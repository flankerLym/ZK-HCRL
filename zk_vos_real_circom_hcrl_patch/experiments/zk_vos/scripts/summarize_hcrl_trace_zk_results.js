const fs = require("fs");
const path = require("path");
function readCsv(file) {
  if (!fs.existsSync(file)) return [];
  const lines = fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const header = lines[0].split(",");
  return lines.slice(1).map(l => {
    const vals = l.split(",");
    const o = {}; header.forEach((h, i) => o[h] = vals[i]); return o;
  });
}
function parseGasTxt(s) {
  const out = {};
  const patterns = {
    groth16_deploy_gas: /Groth16Verifier deployment gasUsed:\s*(\d+)/,
    registry_deploy_gas: /OracleScheduleRegistry deployment gasUsed:\s*(\d+)/,
    submit_schedule_real_verifier_gas: /submitSchedule with real Groth16 verifier gasUsed:\s*(\d+)/,
  };
  for (const [k, re] of Object.entries(patterns)) {
    const m = s.match(re); if (m) out[k] = Number(m[1]);
  }
  return out;
}
const base = path.join(__dirname, "..");
const results = path.join(base, "results");
const batchSummary = readCsv(path.join(results, "hcrl_trace_batch_summary.csv"));
const proofRows = readCsv(path.join(results, "hcrl_trace_proof_timing.csv"));
let gas = {};
const gasFile = path.join(results, "hcrl_trace_real_registry_gas.txt");
if (fs.existsSync(gasFile)) gas = parseGasTxt(fs.readFileSync(gasFile, "utf8"));
const proofMean = proofRows.length ? {
  proof_samples: proofRows.length,
  mean_witness_ms: proofRows.reduce((a, r) => a + Number(r.witness_ms || 0), 0) / proofRows.length,
  mean_prove_ms: proofRows.reduce((a, r) => a + Number(r.prove_ms || 0), 0) / proofRows.length,
  mean_verify_ms: proofRows.reduce((a, r) => a + Number(r.verify_ms || 0), 0) / proofRows.length,
  mean_total_ms: proofRows.reduce((a, r) => a + Number(r.total_ms || 0), 0) / proofRows.length,
} : {};
const summary = { createdAt: new Date().toISOString(), batch_validation: batchSummary, proof_timing: proofMean, gas };
fs.writeFileSync(path.join(results, "hcrl_trace_zk_final_summary.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));
console.log(`[DONE] Final summary written to ${path.join(results, "hcrl_trace_zk_final_summary.json")}`);
