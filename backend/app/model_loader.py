# backend/app/model_loader.py
"""
Model loader and feature extractor (robust to feature-size mismatches).

Changes:
- When scoring, we build the canonical 7-element feature vector in a fixed order.
- If the loaded model expects a different number of features, we automatically
  trim (drop from end) or pad with zeros to match model.n_features_in_.
- This makes the backend tolerant to models trained earlier with fewer features.

Features order (canonical):
  0 num_inputs
  1 num_outputs
  2 total_ada
  3 metadata_bytes
  4 has_nft_metadata
  5 metadata_key_count
  6 suspicious_values
"""

import os
import io
import json
import logging
import joblib
import math
import numpy as np
from typing import Dict, Any

from sklearn.ensemble import IsolationForest

import requests

logger = logging.getLogger("model-loader")

MODEL_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
MODEL_PATH = os.getenv("MODEL_PATH", "models/isolation_model.joblib")  # path within Supabase bucket

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        # attempt to load immediately
        try:
            self.load_or_train()
        except Exception as e:
            logger.exception("Initial model load/train failed: %s", e)

    def load_or_train(self):
        # Try Supabase
        if SUPABASE_URL and SUPABASE_SERVICE_KEY:
            try:
                self._download_from_supabase()
                self.model = joblib.load(MODEL_LOCAL_PATH)
                self.model_loaded = True
                logger.info("Loaded model from Supabase into %s", MODEL_LOCAL_PATH)
                return
            except Exception as e:
                logger.warning("Supabase model download/load failed: %s", e)

        # Try local file
        if os.path.exists(MODEL_LOCAL_PATH):
            try:
                self.model = joblib.load(MODEL_LOCAL_PATH)
                self.model_loaded = True
                logger.info("Loaded local model %s", MODEL_LOCAL_PATH)
                return
            except Exception as e:
                logger.warning("Failed to load local model: %s", e)

        # Fallback: quick train (canonical 7-feature model)
        logger.info("No model found — training a tiny IsolationForest as fallback")
        self._train_quick_model()
        self.model_loaded = True

    def _download_from_supabase(self):
        if not MODEL_PATH or "/" not in MODEL_PATH:
            raise ValueError("MODEL_PATH must be 'bucket/path/to/file'")

        bucket, *path_parts = MODEL_PATH.split("/")
        path = "/".join(path_parts)
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
        logger.info("Trying Supabase public URL: %s", public_url)
        r = requests.get(public_url, timeout=15)
        if r.status_code == 200:
            with open(MODEL_LOCAL_PATH, "wb") as fh:
                fh.write(r.content)
            return

        obj_url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
        headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        r2 = requests.get(obj_url, headers=headers, timeout=15)
        if r2.status_code == 200:
            with open(MODEL_LOCAL_PATH, "wb") as fh:
                fh.write(r2.content)
            return

        raise RuntimeError(f"Failed to download model from Supabase (status {r.status_code}, {r2.status_code})")

    def _train_quick_model(self):
        # Train a canonical 7-feature IsolationForest
        rng = np.random.RandomState(42)
        X_normal = []
        for _ in range(200):
            num_inputs = rng.poisson(2) + 1
            num_outputs = rng.poisson(2) + 1
            total_ada = abs(rng.normal(loc=1e6, scale=5e5))
            metadata_bytes = rng.poisson(50)
            has_nft_meta = rng.choice([0, 1], p=[0.9, 0.1])
            meta_keys = rng.poisson(1)
            suspicious_values = 0
            X_normal.append([num_inputs, num_outputs, total_ada, metadata_bytes, has_nft_meta, meta_keys, suspicious_values])
        X_normal = np.array(X_normal)
        clf = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        clf.fit(X_normal)
        joblib.dump(clf, MODEL_LOCAL_PATH)
        self.model = clf
        logger.info("Trained quick IsolationForest (7-feat) and saved to %s", MODEL_LOCAL_PATH)

    def extract_features_from_tx(self, tx: Dict[str, Any]) -> Dict[str, float]:
        inputs = tx.get("inputs", []) or []
        outputs = tx.get("outputs", []) or []
        num_inputs = len(inputs)
        num_outputs = len(outputs)
        total = 0.0
        for out in outputs:
            for amt in out.get("amount", []):
                if amt.get("unit") == "lovelace":
                    try:
                        total += float(amt.get("quantity", 0))
                    except Exception:
                        pass
        metadata = tx.get("metadata", {}) or {}
        metadata_bytes = int(tx.get("metadata_bytes", 0) or 0)
        metadata_key_count = len(metadata.keys())
        has_nft_metadata = 1 if any(k in ("721", "721Policy", "assets") or str(k).startswith("721") for k in metadata.keys()) else 0
        suspicious_values = 0
        for v in metadata.values():
            try:
                sval = json.dumps(v)
                if len(sval) > 200:
                    suspicious_values += 1
            except Exception:
                pass

        features = {
            "num_inputs": float(num_inputs),
            "num_outputs": float(num_outputs),
            "total_ada": float(total),
            "metadata_bytes": float(metadata_bytes),
            "has_nft_metadata": float(has_nft_metadata),
            "metadata_key_count": float(metadata_key_count),
            "suspicious_values": float(suspicious_values),
        }
        return features

    def _build_canonical_vector(self, features: Dict[str, float]) -> list:
        """
        Return canonical ordered list of 7 features:
        [num_inputs, num_outputs, total_ada, metadata_bytes, has_nft_metadata, metadata_key_count, suspicious_values]
        """
        order = ["num_inputs", "num_outputs", "total_ada", "metadata_bytes", "has_nft_metadata", "metadata_key_count", "suspicious_values"]
        vec = [ float(features.get(k, 0.0)) for k in order ]
        return vec

    def _align_vector_to_model(self, vec: list) -> np.ndarray:
        """
        Ensure the vector length matches model.n_features_in_.
        - If model expects fewer features, trim from the end.
        - If model expects more features, pad with zeros.
        - If model is None, return as numpy array.
        """
        arr = np.array(vec, dtype=float).reshape(1, -1)
        if self.model is None:
            return arr

        expected = getattr(self.model, "n_features_in_", None)
        if expected is None:
            # unknown - assume arr is fine
            return arr

        current = arr.shape[1]
        if current == expected:
            return arr
        if current > expected:
            logger.warning("Model expects %d features but input has %d: trimming extra features", expected, current)
            return arr[:, :expected]
        # pad with zeros
        pad_width = expected - current
        logger.warning("Model expects %d features but input has %d: padding with %d zeros", expected, current, pad_width)
        pad = np.zeros((1, pad_width), dtype=float)
        return np.concatenate([arr, pad], axis=1)

    def score_features(self, features: Dict[str, float]):
        """
        Build canonical vector, align to model, then score.
        Returns (anomaly_score 0..1, label)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        canonical = self._build_canonical_vector(features)
        X = self._align_vector_to_model(canonical)

        # IsolationForest.decision_function: higher = more normal
        df = self.model.decision_function(X)[0]
        score = 1.0 / (1.0 + math.exp(df))
        score = max(0.0, min(1.0, score))
        if score < 0.30:
            label = "LOW"
        elif score < 0.60:
            label = "MEDIUM"
        else:
            label = "HIGH"
        return round(score, 4), label
