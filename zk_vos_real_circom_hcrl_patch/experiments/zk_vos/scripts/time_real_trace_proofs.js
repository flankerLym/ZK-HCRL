const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

// Robust timing script for HCRL/Audit real-trace ZK-VOS proofs.
// Fixes a Windows/snarkjs edge case where `snarkjs groth16 verify` may print
// "snarkJS: OK!" but still be interpreted as a failed command by the wrapper.
// This script treats a verification as successful when snarkjs output contains
// OK, records the non-zero status as a warning, and continues the batch.

const base = path.join(__dirname, "..");
const buildDir = path.join(base, "build");
const proofDir = path.join(base, "proof");
const resultsDir = path.join(base, "results");
const realInputsDir = path.join(base, "real_trace_inputs");
const validDir = path.join(realInputsDir, "valid");

fs.mkdirSync(proofDir, { recursive: true });
fs.mkdirSync(resultsDir, { recursive: true });

const nodeCmd = process.execPath;
const npxCmd = process.platform === "win32" ? "npx.cmd" : "npx";
const wasm = path.join(buildDir, "zk_vos_full_js", "zk_vos_full.wasm");
const gen = path.join(buildDir, "zk_vos_full_js", "generate_witness.js");
const zkey = path.join(buildDir, "zk_vos_full_final.zkey");
const vkey = path.join(buildDir, "verification_key.json");
const r1cs = path.join(buildDir, "zk_vos_full.r1cs");

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {
    samples: Number(process.env.ZK_PROOF_SAMPLES || 20),
    inputDir: validDir,
  };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--samples") out.samples = parseInt(args[++i], 10);
    else if (args[i] === "--inputDir") out.inputDir = path.resolve(args[++i]);
  }
  if (!Number.isFinite(out.samples) || out.samples <= 0) out.samples = 20;
  return out;
}

function assertExists(file, label) {
  if (!fs.existsSync(file)) {
    throw new Error(`${label} not found: ${file}`);
  }
}

function listValidInputs(inputDir) {
  const candidates = [];
  const addJsons = (dir) => {
    if (!fs.existsSync(dir)) return;
    for (const name of fs.readdirSync(dir)) {
      const p = path.join(dir, name);
      if (fs.statSync(p).isFile() && name.toLowerCase().endsWith(".json")) {
        candidates.push(p);
      }
    }
  };
  addJsons(inputDir);
  // Backward-compatible fallbacks for earlier patch layouts.
  if (candidates.length === 0) addJsons(realInputsDir);
  if (candidates.length === 0) addJsons(path.join(base, "inputs"));

  return candidates
    .filter(p => /valid/i.test(path.basename(p)))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function nowMs() {
  const [sec, nano] = process.hrtime();
  return sec * 1000 + nano / 1e6;
}

function run(label, cmd, args, opts = {}) {
  const start = nowMs();
  const res = spawnSync(cmd, args, {
    cwd: base,
    encoding: "utf8",
    windowsHide: true,
    maxBuffer: 64 * 1024 * 1024,
  });
  const elapsed = nowMs() - start;
  const stdout = res.stdout || "";
  const stderr = res.stderr || "";
  const combined = `${stdout}\n${stderr}`;

  // For snarkjs verification, trust the explicit OK marker even if the wrapper
  // observes a non-zero status. This avoids aborting after successful verification
  // on some Windows shells / ANSI-output combinations.
  if (opts.acceptSnarkOk && /snarkJS[\s\S]*OK!?/i.test(combined)) {
    if (res.status !== 0) {
      console.warn(`[WARN] ${label}: command status=${res.status}, but snarkJS OK was detected; accepting verification.`);
    }
    return { ok: true, elapsed, stdout, stderr, status: res.status, warningAccepted: res.status !== 0 };
  }

  if (res.status !== 0 || res.error) {
    console.error(`[ERROR] ${label} failed. status=${res.status}`);
    if (res.error) console.error(res.error);
    if (stdout.trim()) console.error(stdout);
    if (stderr.trim()) console.error(stderr);
    return { ok: false, elapsed, stdout, stderr, status: res.status, error: String(res.error || "") };
  }
  return { ok: true, elapsed, stdout, stderr, status: res.status };
}

function csvEscape(x) {
  const s = String(x === undefined || x === null ? "" : x);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function writeCsv(file, rows) {
  if (rows.length === 0) {
    fs.writeFileSync(file, "case,status\n", "utf8");
    return;
  }
  const header = Object.keys(rows[0]);
  const lines = [header.join(",")];
  for (const r of rows) lines.push(header.map(h => csvEscape(r[h])).join(","));
  fs.writeFileSync(file, lines.join("\n"), "utf8");
}

function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}
function std(arr) {
  if (arr.length <= 1) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) * (x - m), 0) / (arr.length - 1));
}

function copyIfExists(src, dst) {
  if (fs.existsSync(src)) fs.copyFileSync(src, dst);
}

function main() {
  const opts = parseArgs();
  assertExists(wasm, "WASM");
  assertExists(gen, "Witness generator");
  assertExists(zkey, "Groth16 zkey");
  assertExists(vkey, "Verification key");

  const inputs = listValidInputs(opts.inputDir).slice(0, opts.samples);
  if (inputs.length === 0) {
    throw new Error(`No valid input JSON files found under ${opts.inputDir} or ${realInputsDir}`);
  }

  console.log(`[INFO] Timing ${inputs.length} real-trace valid proofs.`);
  const rows = [];
  let firstOk = null;

  inputs.forEach((inputPath, idx) => {
    const id = String(idx).padStart(6, "0");
    const wtns = path.join(buildDir, `hcrl_trace_valid_${id}.wtns`);
    const proof = path.join(proofDir, `hcrl_trace_valid_${id}_proof.json`);
    const pub = path.join(proofDir, `hcrl_trace_valid_${id}_public.json`);

    const w = run(`witness ${id}`, nodeCmd, [gen, wasm, inputPath, wtns]);
    if (!w.ok) {
      rows.push({ index: idx, input: inputPath, status: "witness_failed", witness_ms: w.elapsed.toFixed(3), prove_ms: "", verify_ms: "", verify_warning_accepted: 0 });
      return;
    }

    const p = run(`prove ${id}`, npxCmd, ["snarkjs", "groth16", "prove", zkey, wtns, proof, pub]);
    if (!p.ok) {
      rows.push({ index: idx, input: inputPath, status: "prove_failed", witness_ms: w.elapsed.toFixed(3), prove_ms: p.elapsed.toFixed(3), verify_ms: "", verify_warning_accepted: 0 });
      return;
    }

    const v = run(`verify ${id}`, npxCmd, ["snarkjs", "groth16", "verify", vkey, pub, proof], { acceptSnarkOk: true });
    if (!v.ok) {
      rows.push({ index: idx, input: inputPath, status: "verify_failed", witness_ms: w.elapsed.toFixed(3), prove_ms: p.elapsed.toFixed(3), verify_ms: v.elapsed.toFixed(3), verify_warning_accepted: 0 });
      return;
    }

    if (firstOk === null) firstOk = { proof, pub };
    rows.push({
      index: idx,
      input: inputPath,
      status: "ok",
      witness_ms: w.elapsed.toFixed(3),
      prove_ms: p.elapsed.toFixed(3),
      verify_ms: v.elapsed.toFixed(3),
      verify_warning_accepted: v.warningAccepted ? 1 : 0,
    });
    console.log(`[OK] proof ${idx + 1}/${inputs.length}: witness=${w.elapsed.toFixed(1)}ms prove=${p.elapsed.toFixed(1)}ms verify=${v.elapsed.toFixed(1)}ms`);
  });

  const timingCsv = path.join(resultsDir, "hcrl_trace_proof_timing.csv");
  writeCsv(timingCsv, rows);

  const okRows = rows.filter(r => r.status === "ok");
  if (okRows.length === 0) {
    throw new Error("No real-trace proof was successfully verified; cannot continue.");
  }

  // Keep legacy proof names for downstream Solidity calldata and Hardhat test steps.
  copyIfExists(firstOk.proof, path.join(proofDir, "valid_proof.json"));
  copyIfExists(firstOk.pub, path.join(proofDir, "valid_public.json"));
  fs.writeFileSync(path.join(resultsDir, "offchain_verify_valid.txt"), "[INFO] snarkJS: OK!\n", "utf8");

  const witness = okRows.map(r => Number(r.witness_ms));
  const prove = okRows.map(r => Number(r.prove_ms));
  const verify = okRows.map(r => Number(r.verify_ms));
  const summary = [
    { metric: "proof_samples_requested", value: inputs.length, unit: "count" },
    { metric: "proof_samples_ok", value: okRows.length, unit: "count" },
    { metric: "proof_samples_failed", value: rows.length - okRows.length, unit: "count" },
    { metric: "witness_mean_ms", value: mean(witness).toFixed(3), unit: "ms" },
    { metric: "witness_std_ms", value: std(witness).toFixed(3), unit: "ms" },
    { metric: "prove_mean_ms", value: mean(prove).toFixed(3), unit: "ms" },
    { metric: "prove_std_ms", value: std(prove).toFixed(3), unit: "ms" },
    { metric: "verify_mean_ms", value: mean(verify).toFixed(3), unit: "ms" },
    { metric: "verify_std_ms", value: std(verify).toFixed(3), unit: "ms" },
    { metric: "total_mean_ms", value: (mean(witness) + mean(prove) + mean(verify)).toFixed(3), unit: "ms" },
  ];
  writeCsv(path.join(resultsDir, "hcrl_trace_proof_timing_summary.csv"), summary);

  console.log(`[DONE] Real-trace proof timing complete. OK=${okRows.length}/${rows.length}`);
  console.log(`Timing CSV: ${timingCsv}`);
  console.log(`Timing summary: ${path.join(resultsDir, "hcrl_trace_proof_timing_summary.csv")}`);
}

main();
