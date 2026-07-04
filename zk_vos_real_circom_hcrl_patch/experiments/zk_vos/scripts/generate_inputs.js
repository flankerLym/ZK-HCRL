const fs = require("fs");
const path = require("path");
const circomlibjs = require("circomlibjs");

// ZK-VOS input generator.
// It reads:
//   data/sample_oracle_pool.json
//   data/sample_schedules.csv
// and emits valid/invalid circuit inputs under inputs/.
// This is the bridge between HCRL/Audit outputs and the Circom proof.

const DEFAULT_LEVELS = 8;
const ZERO = 0n;

function parseArgs() {
  const args = process.argv.slice(2);
  const out = { row: 0, levels: DEFAULT_LEVELS };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--row") out.row = parseInt(args[++i], 10);
    else if (a === "--levels") out.levels = parseInt(args[++i], 10);
    else if (a === "--pool") out.pool = args[++i];
    else if (a === "--schedules") out.schedules = args[++i];
    else if (a === "--out") out.out = args[++i];
  }
  return out;
}

function toStr(x) { return BigInt(x).toString(); }

function readCsv(file) {
  const raw = fs.readFileSync(file, "utf8").trim();
  const lines = raw.split(/\r?\n/).filter(Boolean);
  const header = lines[0].split(",").map(s => s.trim());
  return lines.slice(1).map(line => {
    const vals = line.split(",").map(s => s.trim());
    const obj = {};
    header.forEach((h, idx) => obj[h] = vals[idx]);
    return obj;
  });
}

function getFirst(obj, names, fallback = undefined) {
  for (const n of names) {
    if (obj[n] !== undefined && obj[n] !== "") return obj[n];
  }
  if (fallback !== undefined) return fallback;
  throw new Error(`Missing one of fields: ${names.join(", ")}`);
}

async function main() {
  const opts = parseArgs();
  const LEVELS = opts.levels;
  const baseDir = path.join(__dirname, "..");
  const dataDir = path.join(baseDir, "data");
  const inputsDir = opts.out ? path.resolve(opts.out) : path.join(baseDir, "inputs");
  fs.mkdirSync(inputsDir, { recursive: true });

  const poolPath = opts.pool ? path.resolve(opts.pool) : path.join(dataDir, "sample_oracle_pool.json");
  const schedulesPath = opts.schedules ? path.resolve(opts.schedules) : path.join(dataDir, "sample_schedules.csv");

  const poseidon = await circomlibjs.buildPoseidon();
  const F = poseidon.F;

  function poseidonHash(arr) {
    return BigInt(F.toObject(poseidon(arr.map(BigInt))));
  }

  function normalizeOracle(o) {
    return {
      oracleId: Number(getFirst(o, ["oracleId", "oracle_id", "id"])),
      oracleServiceType: Number(getFirst(o, ["oracleServiceType", "oracle_service_type", "serviceType", "service_type"])),
      repEff: Number(getFirst(o, ["repEff", "rep_eff", "effectiveReputation", "effective_reputation", "audit_reputation"])),
      cost: Number(getFirst(o, ["cost", "oracle_cost"])),
      risk: Number(getFirst(o, ["risk", "risk_score", "validationRisk", "validation_risk"])),
      latencyEst: Number(getFirst(o, ["latencyEst", "latency_est", "latency", "response_time_est"])),
      cooldown: Number(getFirst(o, ["cooldown", "cooldown_flag"], 0))
    };
  }

  function normalizeSchedule(s) {
    return {
      requestId: Number(getFirst(s, ["requestId", "request_id", "req_id", "id"])),
      selectedOracleId: Number(getFirst(s, ["selectedOracleId", "selected_oracle_id", "oracle_id", "selected_oracle"])),
      requestServiceType: Number(getFirst(s, ["requestServiceType", "request_service_type", "service_type", "req_type"])),
      reputationThreshold: Number(getFirst(s, ["reputationThreshold", "reputation_threshold", "rep_threshold", "tau_rep"], 7000)),
      costBudget: Number(getFirst(s, ["costBudget", "cost_budget", "budget_cost", "B_cost"], 500)),
      riskBudget: Number(getFirst(s, ["riskBudget", "risk_budget", "budget_risk", "B_risk"], 300)),
      deadline: Number(getFirst(s, ["deadline", "latency_budget", "B_latency"], 120))
    };
  }

  function leafOf(o) {
    return poseidonHash([
      o.oracleId,
      o.oracleServiceType,
      o.repEff,
      o.cost,
      o.risk,
      o.latencyEst,
      o.cooldown
    ]);
  }

  function oracleIdHash(id) {
    return poseidonHash([id]);
  }

  function buildMerkle(leaves, levels) {
    let size = 1 << levels;
    if (leaves.length > size) {
      throw new Error(`Oracle pool has ${leaves.length} leaves but levels=${levels} supports only ${size}`);
    }
    let padded = leaves.slice();
    while (padded.length < size) padded.push(ZERO);
    let layers = [padded];
    for (let lvl = 0; lvl < levels; lvl++) {
      let prev = layers[lvl];
      let next = [];
      for (let i = 0; i < prev.length; i += 2) {
        next.push(poseidonHash([prev[i], prev[i + 1]]));
      }
      layers.push(next);
    }
    return layers;
  }

  function getPath(layers, index, levels) {
    let pathElements = [];
    let pathIndices = [];
    let idx = index;
    for (let lvl = 0; lvl < levels; lvl++) {
      let sibling = idx ^ 1;
      pathElements.push(layers[lvl][sibling]);
      pathIndices.push(idx & 1);
      idx = Math.floor(idx / 2);
    }
    return { pathElements, pathIndices };
  }

  const rawPool = JSON.parse(fs.readFileSync(poolPath, "utf8"));
  const poolList = Array.isArray(rawPool) ? rawPool : rawPool.oracles;
  const pool = poolList.map(normalizeOracle);
  const schedules = readCsv(schedulesPath).map(normalizeSchedule);
  if (!schedules[opts.row]) throw new Error(`No schedule row ${opts.row} in ${schedulesPath}`);
  const schedule = schedules[opts.row];

  const selectedIndex = pool.findIndex(o => o.oracleId === schedule.selectedOracleId);
  if (selectedIndex < 0) throw new Error(`selectedOracleId=${schedule.selectedOracleId} not found in pool`);
  const selected = pool[selectedIndex];

  const leaves = pool.map(leafOf);
  const layers = buildMerkle(leaves, LEVELS);
  const root = layers[LEVELS][0];
  const pathInfo = getPath(layers, selectedIndex, LEVELS);

  function makeInput(overrides = {}) {
    const req = {
      requestId: schedule.requestId,
      selectedOracleHash: oracleIdHash(selected.oracleId),
      oraclePoolRoot: root,
      reputationThreshold: schedule.reputationThreshold,
      costBudget: schedule.costBudget,
      riskBudget: schedule.riskBudget,
      deadline: schedule.deadline,
      requestServiceType: schedule.requestServiceType,
      oracleId: selected.oracleId,
      oracleServiceType: selected.oracleServiceType,
      repEff: selected.repEff,
      cost: selected.cost,
      risk: selected.risk,
      latencyEst: selected.latencyEst,
      cooldown: selected.cooldown,
      pathElements: pathInfo.pathElements,
      pathIndices: pathInfo.pathIndices,
      ...overrides
    };
    const out = {};
    for (const [k, v] of Object.entries(req)) {
      if (Array.isArray(v)) out[k] = v.map(toStr);
      else out[k] = toStr(v);
    }
    return out;
  }

  const files = {
    "valid_schedule.json": makeInput(),
    "invalid_low_reputation.json": makeInput({ repEff: Math.max(0, schedule.reputationThreshold - 1) }),
    "invalid_over_cost.json": makeInput({ cost: schedule.costBudget + 1 }),
    "invalid_over_risk.json": makeInput({ risk: schedule.riskBudget + 1 }),
    "invalid_over_latency.json": makeInput({ latencyEst: schedule.deadline + 1 }),
    "invalid_cooldown.json": makeInput({ cooldown: 1 }),
    "invalid_service_mismatch.json": makeInput({ oracleServiceType: schedule.requestServiceType + 1 }),
    "invalid_membership.json": makeInput({ pathElements: pathInfo.pathElements.map((x, idx) => idx === 0 ? x + 1n : x) })
  };

  for (const [name, obj] of Object.entries(files)) {
    fs.writeFileSync(path.join(inputsDir, name), JSON.stringify(obj, null, 2));
  }

  const meta = {
    row: opts.row,
    levels: LEVELS,
    selectedOracleId: selected.oracleId,
    oraclePoolRoot: root.toString(),
    publicSignalsOrder: [
      "requestId", "selectedOracleHash", "oraclePoolRoot", "reputationThreshold",
      "costBudget", "riskBudget", "deadline", "requestServiceType"
    ],
    selectedOracle: selected,
    schedule
  };
  fs.writeFileSync(path.join(inputsDir, "metadata.json"), JSON.stringify(meta, null, 2));
  fs.writeFileSync(path.join(inputsDir, "oracle_pool_root.txt"), root.toString());
  console.log(`Generated ZK-VOS inputs in ${inputsDir}`);
  console.log(`selectedOracleId=${selected.oracleId}`);
  console.log(`oraclePoolRoot=${root.toString()}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
