// frontend/app.js
// Simple client to call backend endpoints. By default expects backend at same origin under /.
// If backend is separate, set BACKEND_URL environment variable in Vercel or edit below.
const BACKEND_URL = (typeof BACKEND_URL !== "undefined" && BACKEND_URL) || ""; // leave empty => same origin

function $(id){ return document.getElementById(id); }

async function fetchJSON(path){
  const url = (BACKEND_URL || "") + path;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return await res.json();
}

function setStatus(msg){ $("status").textContent = "Status: " + msg; }

function riskClass(label){
  if (label==="LOW") return "low";
  if (label==="MEDIUM") return "med";
  return "high";
}

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
    setStatus("error");
    $("result").textContent = "Error: " + err.message;
  }
});

