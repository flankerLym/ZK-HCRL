const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const base = path.join(__dirname, "..");
const wasm = path.join(base, "build", "zk_vos_full_js", "zk_vos_full.wasm");
const gen = path.join(base, "build", "zk_vos_full_js", "generate_witness.js");
const inputsDir = path.join(base, "inputs");
const resultsDir = path.join(base, "results");
fs.mkdirSync(resultsDir, { recursive: true });

const cases = [
  "invalid_low_reputation.json",
  "invalid_over_cost.json",
  "invalid_over_risk.json",
  "invalid_over_latency.json",
  "invalid_cooldown.json",
  "invalid_service_mismatch.json",
  "invalid_membership.json"
];

const rows = [];
for (const c of cases) {
  const input = path.join(inputsDir, c);
  const outWtns = path.join(base, "build", c.replace(".json", ".wtns"));
  const res = spawnSync("node", [gen, wasm, input, outWtns], { encoding: "utf8" });
  const rejected = res.status !== 0;
  rows.push({ case: c, expected: "rejected", status: rejected ? "rejected" : "UNEXPECTED_ACCEPTED" });
  if (!rejected) {
    console.error(`[FAIL] ${c} unexpectedly generated a witness.`);
  } else {
    console.log(`[OK] ${c} rejected by circuit constraints.`);
  }
}

const csv = ["case,expected,status", ...rows.map(r => `${r.case},${r.expected},${r.status}`)].join("\n");
fs.writeFileSync(path.join(resultsDir, "invalid_witness_checks.csv"), csv);
if (rows.some(r => r.status !== "rejected")) process.exit(1);
