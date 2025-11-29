// frontend/app.js
// Backend URL: default to localhost:8000 (Codespaces). Override by setting window.BACKEND_URL in browser or editing this file.
const BACKEND_URL = (typeof window !== "undefined" && window.BACKEND_URL) ? window.BACKEND_URL : "http://127.0.0.1:8000";

function $(id){ return document.getElementById(id); }

async function fetchJSON(path){
  const url = BACKEND_URL + path;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return await res.json();
}

function setStatus(msg){ $("status").textContent = "Status: " + msg; }

$("btnFetch").addEventListener("click", async () => {
  const tx = $("txhash").value.trim() || "mock_tx_123";
  setStatus("fetching transaction...");
  $("result").textContent = "";
  try {
    setStatus("fetching /cardano/tx/" + tx);
    const txData = await fetchJSON("/cardano/tx/" + encodeURIComponent(tx));
    setStatus("scoring tx...");
    const scoreData = await fetchJSON("/score/tx/" + encodeURIComponent(tx));
    setStatus("rendering result");
    const out = [];
    out.push("TX HASH: " + scoreData.tx_hash);
    out.push("RISK SCORE: " + scoreData.anomaly_score + "  (" + scoreData.risk_label + ")");
    out.push("");
    out.push("Features:");
    for (const k of Object.keys(scoreData.features)) {
      out.push(`  ${k}: ${scoreData.features[k]}`);
    }
    out.push("");
    out.push("Metadata (raw):");
    out.push(JSON.stringify(scoreData.metadata, null, 2));
    $("result").textContent = out.join("\n");
    setStatus("done");
  } catch (err) {
    setStatus("error: " + err.message);
    $("result").textContent = "Error: " + err.message;
    console.error("Fetch error", err);
  }
});
