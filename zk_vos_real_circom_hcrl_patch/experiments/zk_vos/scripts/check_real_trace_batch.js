const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");
const { performance } = require("perf_hooks");

function parseArgs() {
  const args = process.argv.slice(2);
  const out = { inDir: "real_trace_inputs", maxValid: 1000000, maxInvalidPerCase: 1000000 };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--in" || a === "--inDir") out.inDir = args[++i];
    else if (a === "--maxValid") out.maxValid = parseInt(args[++i], 10);
    else if (a === "--maxInvalidPerCase") out.maxInvalidPerCase = parseInt(args[++i], 10);
  }
  return out;
}
function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function listJson(dir, limit) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f => f.endsWith(".json")).sort().slice(0, limit).map(f => path.join(dir, f));
}
function csvEscape(x) {
  const s = String(x === undefined ? "" : x);
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function writeCsv(file, rows) {
  if (!rows.length) { fs.writeFileSync(file, ""); return; }
  const keys = Object.keys(rows[0]);
  fs.writeFileSync(file, [keys.join(","), ...rows.map(r => keys.map(k => csvEscape(r[k])).join(","))].join("\n"));
}
function tail(s, n = 500) { return String(s || "").slice(-n).replace(/\r?\n/g, " | "); }

const opts = parseArgs();
const base = path.join(__dirname, "..");
const inputBase = path.resolve(base, opts.inDir);
const wasm = path.join(base, "build", "zk_vos_full_js", "zk_vos_full.wasm");
const gen = path.join(base, "build", "zk_vos_full_js", "generate_witness.js");
const witnessDir = path.join(base, "build", "hcrl_trace_batch_witnesses");
const resultsDir = path.join(base, "results");
ensureDir(witnessDir);
ensureDir(resultsDir);

if (!fs.existsSync(wasm) || !fs.existsSync(gen)) {
  console.error("[ERROR] Circuit WASM/generate_witness.js not found. Compile first: circom circuits\\zk_vos_full.circom --r1cs --wasm --sym -o build");
  process.exit(1);
}

const rows = [];
function runOne(file, expected, caseType, idx) {
  const outWtns = path.join(witnessDir, `${caseType}_${idx}.wtns`);
  const t0 = performance.now();
  const res = spawnSync("node", [gen, wasm, file, outWtns], { encoding: "utf8", timeout: 120000 });
  const dt = performance.now() - t0;
  const accepted = res.status === 0;
  const observed = accepted ? "accepted" : "rejected";
  const ok = observed === expected;
  rows.push({
    case_type: caseType,
    input_file: path.relative(base, file),
    expected,
    observed,
    ok: ok ? 1 : 0,
    exit_code: res.status,
    time_ms: dt.toFixed(3),
    stderr_tail: tail(res.stderr),
  });
  if (!ok) console.error(`[FAIL] ${caseType} ${path.basename(file)} expected ${expected} but observed ${observed}`);
}

const validFiles = listJson(path.join(inputBase, "valid"), opts.maxValid);
validFiles.forEach((f, i) => runOne(f, "accepted", "valid", i));

const mutBase = path.join(inputBase, "mutations");
const mutationCases = fs.existsSync(mutBase) ? fs.readdirSync(mutBase).filter(f => fs.statSync(path.join(mutBase, f)).isDirectory()).sort() : [];
for (const c of mutationCases) {
  listJson(path.join(mutBase, c), opts.maxInvalidPerCase).forEach((f, i) => runOne(f, "rejected", c, i));
}

writeCsv(path.join(resultsDir, "hcrl_trace_batch_validation.csv"), rows);
const summary = [];
for (const caseType of [...new Set(rows.map(r => r.case_type))]) {
  const sub = rows.filter(r => r.case_type === caseType);
  summary.push({
    case_type: caseType,
    samples: sub.length,
    accepted: sub.filter(r => r.observed === "accepted").length,
    rejected: sub.filter(r => r.observed === "rejected").length,
    expected_pass: sub.filter(r => Number(r.ok) === 1).length,
    accuracy: sub.length ? (sub.filter(r => Number(r.ok) === 1).length / sub.length).toFixed(6) : "0",
    mean_time_ms: sub.length ? (sub.reduce((a, r) => a + Number(r.time_ms), 0) / sub.length).toFixed(3) : "0",
  });
}
writeCsv(path.join(resultsDir, "hcrl_trace_batch_summary.csv"), summary);
fs.writeFileSync(path.join(resultsDir, "hcrl_trace_batch_summary.json"), JSON.stringify({ createdAt: new Date().toISOString(), rows: summary }, null, 2));

console.log(`[DONE] Batch witness validation complete.`);
console.log(`Results: ${path.join(resultsDir, "hcrl_trace_batch_validation.csv")}`);
console.log(`Summary: ${path.join(resultsDir, "hcrl_trace_batch_summary.csv")}`);
if (rows.some(r => Number(r.ok) !== 1)) process.exit(2);
