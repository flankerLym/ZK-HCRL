const fs = require("fs");
const path = require("path");
const circomlibjs = require("circomlibjs");

const DEFAULT_LEVELS = 8;
const ZERO = 0n;

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {
    levels: DEFAULT_LEVELS,
    maxValid: 1000,
    maxPerMutation: 1000,
    out: "real_trace_inputs",
    singleOut: "inputs",
    dataDir: "data",
    requireCompliant: true,
    validMode: "relaxed",
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--levels") out.levels = parseInt(args[++i], 10);
    else if (a === "--maxValid") out.maxValid = parseInt(args[++i], 10);
    else if (a === "--maxPerMutation") out.maxPerMutation = parseInt(args[++i], 10);
    else if (a === "--out") out.out = args[++i];
    else if (a === "--singleOut") out.singleOut = args[++i];
    else if (a === "--dataDir") out.dataDir = args[++i];
    else if (a === "--validMode") out.validMode = String(args[++i]).toLowerCase();
    else if (a === "--schedule" || a === "--schedules") out.schedule = args[++i];
    else if (a === "--snapshot" || a === "--snapshots") out.snapshot = args[++i];
    else if (a === "--allowNonCompliant") out.requireCompliant = false;
    else if (a === "--help" || a === "-h") {
      console.log(`Usage:
node scripts/build_real_trace_inputs.js [options]

Options:
  --dataDir data                         Directory containing HCRL trace files
  --schedule path/to/*hcrl_zk_schedule_trace.csv
  --snapshot path/to/*oracle_pool_snapshot.jsonl
  --levels 8                             Merkle tree depth; circuit currently uses 8
  --maxValid 1000                        Number of valid HCRL rows to export
  --maxPerMutation 1000                  Number of invalid inputs per mutation type
  --out real_trace_inputs                Batch output directory
  --singleOut inputs                     Also emit first valid/mutation inputs to existing inputs/ directory
  --validMode strict|relaxed|candidate   How to obtain valid proof inputs from real traces. Default: relaxed
                                        strict: only rows already compliant under logged thresholds
                                        relaxed: keep selected HCRL oracle but relax public bounds to make the proof valid
                                        candidate: use a compliant oracle from the same pool under logged thresholds
  --allowNonCompliant                    Legacy flag; not recommended for proof generation`);
      process.exit(0);
    }
  }
  return out;
}

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function toStr(x) { return BigInt(x).toString(); }
function clean(v) { return String(v === undefined || v === null ? "" : v).trim().replace(/^"|"$/g, ""); }
function has(obj, key) { return Object.prototype.hasOwnProperty.call(obj, key) && clean(obj[key]) !== ""; }
function firstExisting(obj, names) {
  for (const n of names) if (has(obj, n)) return clean(obj[n]);
  return undefined;
}
function firstNumber(obj, names, fallback = undefined) {
  const v = firstExisting(obj, names);
  if (v === undefined) {
    if (fallback !== undefined) return fallback;
    throw new Error(`Missing numeric field among: ${names.join(", ")}`);
  }
  const num = Number(v);
  if (!Number.isFinite(num)) throw new Error(`Invalid number '${v}' for fields ${names.join(", ")}`);
  return num;
}
function parseBool01(v) {
  if (v === undefined || v === null || String(v).trim() === "") return undefined;
  const s = String(v).trim().toLowerCase();
  if (["1", "true", "yes", "y"].includes(s)) return 1;
  if (["0", "false", "no", "n"].includes(s)) return 0;
  const n = Number(s);
  if (Number.isFinite(n)) return n > 0 ? 1 : 0;
  return undefined;
}
function scaledValue(obj, scaledNames, rawNames, scale, fallback = undefined) {
  const scaled = firstExisting(obj, scaledNames);
  if (scaled !== undefined) {
    const n = Number(scaled);
    if (!Number.isFinite(n)) throw new Error(`Invalid scaled value '${scaled}'`);
    return Math.round(n);
  }
  const raw = firstExisting(obj, rawNames);
  if (raw !== undefined) {
    const n = Number(raw);
    if (!Number.isFinite(n)) throw new Error(`Invalid raw value '${raw}'`);
    // If the value already looks scaled, keep it. This makes the script compatible
    // with traces whose deadline/costBudget columns were intentionally stored scaled.
    if (Math.abs(n) >= scale) return Math.round(n);
    return Math.round(n * scale);
  }
  if (fallback !== undefined) return fallback;
  throw new Error(`Missing scaled/raw value among ${scaledNames.concat(rawNames).join(", ")}`);
}
function csvParseLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') { cur += '"'; i++; }
      else inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}
function readCsv(file) {
  const raw = fs.readFileSync(file, "utf8").replace(/^\uFEFF/, "");
  const lines = raw.split(/\r?\n/).filter(l => l.trim().length > 0);
  if (!lines.length) return [];
  const header = csvParseLine(lines[0]).map(s => clean(s));
  return lines.slice(1).map((line, rowIndex) => {
    const vals = csvParseLine(line);
    const obj = { __rowIndex: rowIndex };
    header.forEach((h, idx) => obj[h] = vals[idx] === undefined ? "" : vals[idx]);
    return obj;
  });
}
function findFirstFile(dataDir, contains, suffix) {
  if (!fs.existsSync(dataDir)) return undefined;
  const files = fs.readdirSync(dataDir)
    .filter(f => f.toLowerCase().includes(contains.toLowerCase()) && f.toLowerCase().endsWith(suffix.toLowerCase()))
    .sort();
  return files.length ? path.join(dataDir, files[0]) : undefined;
}
function normalizeRequestRow(row) {
  const requestId = Math.round(firstNumber(row, ["requestId", "request_id", "req_id", "id", "request"]));
  const selectedOracleId = Math.round(firstNumber(row, ["selectedOracleId", "selected_oracle_id", "selected_oracle", "oracle_id"]));
  const requestServiceType = Math.round(firstNumber(row, ["requestServiceType", "request_service_type", "service_type", "req_type", "request_type"]));
  const reputationThreshold = scaledValue(row, ["reputationThreshold", "reputationThreshold_scaled", "reputation_threshold_scaled", "rep_threshold_scaled", "tau_rep_scaled"], ["reputation_threshold", "rep_threshold", "tau_rep"], 10000, 6000);
  const costBudget = scaledValue(row, ["costBudget", "costBudget_scaled", "cost_budget_scaled", "budget_cost_scaled", "B_cost_scaled"], ["cost_budget", "budget_cost", "B_cost"], 1000, 1000);
  const riskBudget = scaledValue(row, ["riskBudget", "riskBudget_scaled", "risk_budget_scaled", "budget_risk_scaled", "B_risk_scaled"], ["risk_budget", "budget_risk", "B_risk"], 10000, 600);
  const deadline = scaledValue(row, ["deadline_scaled", "latency_budget_scaled", "B_latency_scaled"], ["deadline", "latency_budget", "B_latency"], 1000, 6000);
  const zkFlag = parseBool01(firstExisting(row, ["zk_is_compliant", "is_compliant", "compliant"]));
  return { requestId, selectedOracleId, requestServiceType, reputationThreshold, costBudget, riskBudget, deadline, zkFlag, raw: row };
}
function normalizeOracle(o) {
  const oracleId = Math.round(firstNumber(o, ["oracleId", "oracle_id", "id", "selectedOracleId", "selected_oracle_id"]));
  const oracleServiceType = Math.round(firstNumber(o, ["oracleServiceType", "oracle_service_type", "serviceType", "service_type", "oracleServiceType_scaled"], 0));
  const repEff = scaledValue(o, ["repEff_scaled", "rep_eff_scaled", "effectiveReputation_scaled", "effective_reputation_scaled"], ["repEff", "rep_eff", "effectiveReputation", "effective_reputation", "audit_reputation"], 10000, 0);
  const cost = scaledValue(o, ["cost_scaled", "oracle_cost_scaled"], ["cost", "oracle_cost"], 1000, 0);
  const risk = scaledValue(o, ["risk_scaled", "risk_score_scaled", "validationRisk_scaled", "validation_risk_scaled"], ["risk", "risk_score", "validationRisk", "validation_risk"], 10000, 0);
  const latencyEst = scaledValue(o, ["latencyEst_scaled", "latency_est_scaled", "latency_scaled", "response_time_est_scaled"], ["latencyEst", "latency_est", "latency", "response_time_est"], 1000, 0);
  const cooldownFlag = parseBool01(firstExisting(o, ["cooldown_flag", "cooldownFlag"]));
  const cooldown = cooldownFlag !== undefined ? cooldownFlag : (firstNumber(o, ["cooldown"], 0) > 0 ? 1 : 0);
  return { oracleId, oracleServiceType, repEff, cost, risk, latencyEst, cooldown };
}
function readSnapshots(file) {
  const map = new Map();
  if (!file || !fs.existsSync(file)) return map;
  const lines = fs.readFileSync(file, "utf8").split(/\r?\n/).filter(l => l.trim());
  for (const line of lines) {
    let obj;
    try { obj = JSON.parse(line); } catch (e) { continue; }
    const requestId = Number(obj.request_id ?? obj.requestId ?? obj.id);
    const episode = Number(obj.episode ?? 0);
    const arr = Array.isArray(obj) ? obj : (Array.isArray(obj.oracles) ? obj.oracles : []);
    if (!arr.length || !Number.isFinite(requestId)) continue;
    const pool = arr.map(normalizeOracle);
    map.set(`${episode}|${requestId}`, pool);
    if (!map.has(`*|${requestId}`)) map.set(`*|${requestId}`, pool);
  }
  return map;
}
function poolFor(schedule, snapshotMap) {
  const ep = Number(schedule.raw.episode ?? 0);
  return snapshotMap.get(`${ep}|${schedule.requestId}`) || snapshotMap.get(`*|${schedule.requestId}`);
}
function compliant(schedule, oracle) {
  return oracle.oracleServiceType === schedule.requestServiceType &&
    oracle.repEff >= schedule.reputationThreshold &&
    oracle.cost <= schedule.costBudget &&
    oracle.risk <= schedule.riskBudget &&
    oracle.latencyEst <= schedule.deadline &&
    oracle.cooldown === 0;
}

function cloneSchedule(s) {
  return {
    requestId: s.requestId,
    selectedOracleId: s.selectedOracleId,
    requestServiceType: s.requestServiceType,
    reputationThreshold: s.reputationThreshold,
    costBudget: s.costBudget,
    riskBudget: s.riskBudget,
    deadline: s.deadline,
    zkFlag: s.zkFlag,
    raw: s.raw,
  };
}
function relaxScheduleForOracle(schedule, oracle) {
  const s = cloneSchedule(schedule);
  // Keep the real HCRL-selected oracle and request service type, but use proof-calibrated
  // public bounds so that the compliance circuit can produce a valid proof. This is useful
  // when the training trace was exported with conservative thresholds that mark every row
  // as non-compliant. Rows with service mismatch or non-zero cooldown are still skipped.
  s.reputationThreshold = Math.min(s.reputationThreshold, oracle.repEff);
  s.costBudget = Math.max(s.costBudget, oracle.cost);
  s.riskBudget = Math.max(s.riskBudget, oracle.risk);
  s.deadline = Math.max(s.deadline, oracle.latencyEst);
  return s;
}
function thresholdAdjustment(original, proof, oracle) {
  return {
    request_id: original.requestId,
    selectedOracleId: oracle.oracleId,
    original_zk_flag: original.zkFlag === undefined ? "" : original.zkFlag,
    original_reputationThreshold: original.reputationThreshold,
    proof_reputationThreshold: proof.reputationThreshold,
    original_costBudget: original.costBudget,
    proof_costBudget: proof.costBudget,
    original_riskBudget: original.riskBudget,
    proof_riskBudget: proof.riskBudget,
    original_deadline: original.deadline,
    proof_deadline: proof.deadline,
    oracle_repEff: oracle.repEff,
    oracle_cost: oracle.cost,
    oracle_risk: oracle.risk,
    oracle_latencyEst: oracle.latencyEst,
  };
}
async function makePoseidon() {
  const poseidon = await circomlibjs.buildPoseidon();
  const F = poseidon.F;
  return (arr) => BigInt(F.toObject(poseidon(arr.map(BigInt))));
}
function buildMerkle(leaves, levels, poseidonHash) {
  const size = 1 << levels;
  if (leaves.length > size) throw new Error(`Oracle pool has ${leaves.length} leaves but levels=${levels} supports only ${size}`);
  const padded = leaves.slice();
  while (padded.length < size) padded.push(ZERO);
  const layers = [padded];
  for (let lvl = 0; lvl < levels; lvl++) {
    const prev = layers[lvl];
    const next = [];
    for (let i = 0; i < prev.length; i += 2) next.push(poseidonHash([prev[i], prev[i + 1]]));
    layers.push(next);
  }
  return layers;
}
function getPath(layers, index, levels) {
  const pathElements = [];
  const pathIndices = [];
  let idx = index;
  for (let lvl = 0; lvl < levels; lvl++) {
    const sibling = idx ^ 1;
    pathElements.push(layers[lvl][sibling]);
    pathIndices.push(idx & 1);
    idx = Math.floor(idx / 2);
  }
  return { pathElements, pathIndices };
}
function leafOf(o, poseidonHash) {
  return poseidonHash([o.oracleId, o.oracleServiceType, o.repEff, o.cost, o.risk, o.latencyEst, o.cooldown]);
}
function oracleIdHash(id, poseidonHash) { return poseidonHash([id]); }
function makeCircuitInput(schedule, oracle, pool, index, levels, poseidonHash, overrides = {}) {
  const leaves = pool.map(o => leafOf(o, poseidonHash));
  const layers = buildMerkle(leaves, levels, poseidonHash);
  const root = layers[levels][0];
  const pathInfo = getPath(layers, index, levels);
  const req = {
    requestId: schedule.requestId,
    selectedOracleHash: oracleIdHash(oracle.oracleId, poseidonHash),
    oraclePoolRoot: root,
    reputationThreshold: schedule.reputationThreshold,
    costBudget: schedule.costBudget,
    riskBudget: schedule.riskBudget,
    deadline: schedule.deadline,
    requestServiceType: schedule.requestServiceType,
    oracleId: oracle.oracleId,
    oracleServiceType: oracle.oracleServiceType,
    repEff: oracle.repEff,
    cost: oracle.cost,
    risk: oracle.risk,
    latencyEst: oracle.latencyEst,
    cooldown: oracle.cooldown,
    pathElements: pathInfo.pathElements,
    pathIndices: pathInfo.pathIndices,
    ...overrides,
  };
  const out = {};
  for (const [k, v] of Object.entries(req)) out[k] = Array.isArray(v) ? v.map(toStr) : toStr(v);
  return out;
}
function findCandidate(pool, schedule, kind) {
  function otherOk(o) {
    if (kind !== "service_mismatch" && o.oracleServiceType !== schedule.requestServiceType) return false;
    if (kind !== "low_reputation" && o.repEff < schedule.reputationThreshold) return false;
    if (kind !== "over_cost" && o.cost > schedule.costBudget) return false;
    if (kind !== "over_risk" && o.risk > schedule.riskBudget) return false;
    if (kind !== "over_latency" && o.latencyEst > schedule.deadline) return false;
    if (kind !== "cooldown" && o.cooldown !== 0) return false;
    return true;
  }
  for (let i = 0; i < pool.length; i++) {
    const o = pool[i];
    const target =
      (kind === "low_reputation" && o.repEff < schedule.reputationThreshold) ||
      (kind === "over_cost" && o.cost > schedule.costBudget) ||
      (kind === "over_risk" && o.risk > schedule.riskBudget) ||
      (kind === "over_latency" && o.latencyEst > schedule.deadline) ||
      (kind === "cooldown" && o.cooldown !== 0) ||
      (kind === "service_mismatch" && o.oracleServiceType !== schedule.requestServiceType);
    if (target && otherOk(o)) return { oracle: o, index: i, strategy: "pool_candidate" };
  }
  return null;
}
function writeJson(file, obj) { fs.writeFileSync(file, JSON.stringify(obj, null, 2)); }
function csvEscape(x) {
  const s = String(x === undefined ? "" : x);
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function writeCsv(file, rows) {
  if (!rows.length) { fs.writeFileSync(file, ""); return; }
  const keys = Object.keys(rows[0]);
  fs.writeFileSync(file, [keys.join(","), ...rows.map(r => keys.map(k => csvEscape(r[k])).join(","))].join("\n"));
}

async function main() {
  const opts = parseArgs();
  if (!["strict", "relaxed", "candidate"].includes(opts.validMode)) {
    throw new Error(`Unsupported --validMode ${opts.validMode}. Use strict, relaxed, or candidate.`);
  }
  const base = path.join(__dirname, "..");
  const dataDir = path.resolve(base, opts.dataDir);
  const schedulePath = opts.schedule ? path.resolve(opts.schedule) : findFirstFile(dataDir, "hcrl_zk_schedule_trace", ".csv");
  const snapshotPath = opts.snapshot ? path.resolve(opts.snapshot) : findFirstFile(dataDir, "oracle_pool_snapshot", ".jsonl");
  if (!schedulePath) throw new Error(`No *hcrl_zk_schedule_trace.csv found under ${dataDir}. Put your trace CSV in experiments/zk_vos/data/`);
  if (!snapshotPath) throw new Error(`No *oracle_pool_snapshot.jsonl found under ${dataDir}. Put your snapshot JSONL in experiments/zk_vos/data/`);

  const outDir = path.resolve(base, opts.out);
  const validDir = path.join(outDir, "valid");
  const mutationBase = path.join(outDir, "mutations");
  const singleDir = opts.singleOut ? path.resolve(base, opts.singleOut) : null;
  fs.rmSync(outDir, { recursive: true, force: true });
  ensureDir(validDir);
  ensureDir(mutationBase);
  if (singleDir) { fs.rmSync(singleDir, { recursive: true, force: true }); ensureDir(singleDir); }

  const poseidonHash = await makePoseidon();
  const rows = readCsv(schedulePath).map(normalizeRequestRow);
  const snapshots = readSnapshots(snapshotPath);
  const mutations = ["low_reputation", "over_cost", "over_risk", "over_latency", "cooldown", "service_mismatch", "invalid_membership"];
  mutations.forEach(m => ensureDir(path.join(mutationBase, m)));

  const validRows = [];
  const manifestRows = [];
  const skipRows = [];
  const adjustmentRows = [];
  let singleWritten = false;

  for (const schedule of rows) {
    if (validRows.length >= opts.maxValid) break;
    const pool = poolFor(schedule, snapshots);
    if (!pool) { skipRows.push({ request_id: schedule.requestId, reason: "missing_snapshot" }); continue; }
    const selectedIndex = pool.findIndex(o => o.oracleId === schedule.selectedOracleId);
    if (selectedIndex < 0) { skipRows.push({ request_id: schedule.requestId, reason: "selected_oracle_not_in_snapshot" }); continue; }
    let selected = pool[selectedIndex];
    let proofSchedule = schedule;
    let proofStrategy = "strict_real_compliant";
    const recomputedOk = compliant(schedule, selected);

    if (opts.validMode === "strict") {
      if (opts.requireCompliant && (schedule.zkFlag === 0 || !recomputedOk)) {
        skipRows.push({ request_id: schedule.requestId, reason: `strict_non_compliant_trace_or_recomputed_${schedule.zkFlag}_${recomputedOk}` });
        continue;
      }
    } else if (opts.validMode === "candidate") {
      let candidate = recomputedOk ? { oracle: selected, index: selectedIndex } : null;
      if (!candidate) {
        for (let i = 0; i < pool.length; i++) {
          if (compliant(schedule, pool[i])) { candidate = { oracle: pool[i], index: i }; break; }
        }
      }
      if (!candidate) {
        skipRows.push({ request_id: schedule.requestId, reason: "candidate_mode_no_compliant_oracle_under_logged_thresholds" });
        continue;
      }
      selected = candidate.oracle;
      proofStrategy = candidate.index === selectedIndex ? "candidate_selected_already_compliant" : "candidate_replaced_selected_oracle";
    } else if (opts.validMode === "relaxed") {
      if (selected.oracleServiceType !== schedule.requestServiceType) {
        skipRows.push({ request_id: schedule.requestId, reason: "relaxed_skip_service_mismatch" });
        continue;
      }
      if (selected.cooldown !== 0) {
        skipRows.push({ request_id: schedule.requestId, reason: "relaxed_skip_cooldown_nonzero" });
        continue;
      }
      proofSchedule = relaxScheduleForOracle(schedule, selected);
      proofStrategy = recomputedOk && schedule.zkFlag !== 0 ? "strict_real_compliant" : "selected_threshold_relaxed";
      adjustmentRows.push(thresholdAdjustment(schedule, proofSchedule, selected));
    }

    // The selected oracle index may change in candidate mode.
    const proofSelectedIndex = pool.findIndex(o => o.oracleId === selected.oracleId);
    if (proofSelectedIndex < 0) { skipRows.push({ request_id: schedule.requestId, reason: "internal_selected_index_missing" }); continue; }
    if (!compliant(proofSchedule, selected)) {
      skipRows.push({ request_id: schedule.requestId, reason: `proof_schedule_still_non_compliant_${proofStrategy}` });
      continue;
    }

    const idx = validRows.length;
    const id = String(idx).padStart(6, "0");
    const validInput = makeCircuitInput(proofSchedule, selected, pool, proofSelectedIndex, opts.levels, poseidonHash);
    const validFile = path.join(validDir, `valid_${id}.json`);
    writeJson(validFile, validInput);
    validRows.push({ schedule: proofSchedule, originalSchedule: schedule, pool, selected, selectedIndex: proofSelectedIndex, validFile, proofStrategy });
    manifestRows.push({ kind: "valid", case_type: "valid", index: idx, request_id: schedule.requestId, selectedOracleId: selected.oracleId, file: path.relative(base, validFile), strategy: proofStrategy });

    for (const m of mutations) {
      const mDir = path.join(mutationBase, m);
      const mId = String(idx).padStart(6, "0");
      let input, strategy, mOracle = selected, mIndex = proofSelectedIndex;
      if (m === "invalid_membership") {
        const badPath = makeCircuitInput(proofSchedule, selected, pool, proofSelectedIndex, opts.levels, poseidonHash);
        badPath.pathElements[0] = (BigInt(badPath.pathElements[0]) + 1n).toString();
        input = badPath; strategy = "tampered_merkle_path";
      } else {
        const cand = findCandidate(pool, proofSchedule, m);
        if (cand) { mOracle = cand.oracle; mIndex = cand.index; strategy = cand.strategy; }
        else {
          strategy = "selected_witness_mutation_fallback";
          const overrides = {};
          if (m === "low_reputation") overrides.repEff = Math.max(0, proofSchedule.reputationThreshold - 1);
          else if (m === "over_cost") overrides.cost = proofSchedule.costBudget + 1;
          else if (m === "over_risk") overrides.risk = proofSchedule.riskBudget + 1;
          else if (m === "over_latency") overrides.latencyEst = proofSchedule.deadline + 1;
          else if (m === "cooldown") overrides.cooldown = 1;
          else if (m === "service_mismatch") overrides.oracleServiceType = proofSchedule.requestServiceType + 1;
          input = makeCircuitInput(proofSchedule, selected, pool, proofSelectedIndex, opts.levels, poseidonHash, overrides);
        }
        if (!input) input = makeCircuitInput(proofSchedule, mOracle, pool, mIndex, opts.levels, poseidonHash);
      }
      const mFile = path.join(mDir, `invalid_${m}_${mId}.json`);
      writeJson(mFile, input);
      manifestRows.push({ kind: "invalid", case_type: m, index: idx, request_id: schedule.requestId, selectedOracleId: mOracle.oracleId, file: path.relative(base, mFile), strategy });
    }

    if (!singleWritten && singleDir) {
      writeJson(path.join(singleDir, "valid_schedule.json"), validInput);
      for (const m of mutations) {
        const src = path.join(mutationBase, m, `invalid_${m}_${id}.json`);
        const legacyName = m === "low_reputation" ? "invalid_low_reputation.json" :
          m === "over_cost" ? "invalid_over_cost.json" :
          m === "over_risk" ? "invalid_over_risk.json" :
          m === "over_latency" ? "invalid_over_latency.json" :
          m === "cooldown" ? "invalid_cooldown.json" :
          m === "service_mismatch" ? "invalid_service_mismatch.json" : "invalid_membership.json";
        fs.copyFileSync(src, path.join(singleDir, legacyName));
      }
      writeJson(path.join(singleDir, "metadata.json"), { source: "hcrl_real_trace", validMode: opts.validMode, proofStrategy, schedulePath, snapshotPath, request_id: schedule.requestId, selectedOracleId: selected.oracleId, levels: opts.levels });
      singleWritten = true;
    }
  }

  writeCsv(path.join(outDir, "manifest.csv"), manifestRows);
  writeCsv(path.join(outDir, "skipped_rows.csv"), skipRows);
  writeCsv(path.join(outDir, "threshold_adjustments.csv"), adjustmentRows);
  writeJson(path.join(outDir, "manifest.json"), {
    createdAt: new Date().toISOString(),
    schedulePath, snapshotPath, levels: opts.levels, validMode: opts.validMode,
    validCount: validRows.length,
    mutationCases: mutations,
    mutationInputs: Object.fromEntries(mutations.map(m => [m, Math.min(validRows.length, opts.maxPerMutation)])),
    skippedCount: skipRows.length,
    outDir,
    singleDir,
  });

  console.log(`[DONE] Built HCRL real-trace ZK inputs.`);
  console.log(`Schedule trace: ${schedulePath}`);
  console.log(`Oracle snapshots: ${snapshotPath}`);
  console.log(`Valid inputs: ${validRows.length}`);
  console.log(`Valid mode: ${opts.validMode}`);
  if (adjustmentRows.length) console.log(`Threshold-adjusted valid rows: ${adjustmentRows.length}`);
  console.log(`Skipped rows: ${skipRows.length}`);
  console.log(`Batch inputs: ${outDir}`);
  if (singleDir) console.log(`Legacy single-case inputs for existing pipeline: ${singleDir}`);
  if (validRows.length === 0) process.exit(2);
}

main().catch(err => { console.error(err); process.exit(1); });
