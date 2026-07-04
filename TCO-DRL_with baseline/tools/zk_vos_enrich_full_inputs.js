#!/usr/bin/env node
/*
Generate zk_vos_full.circom-compatible input JSON files.

This helper computes Poseidon values with circomlibjs:
  selectedOracleHash = Poseidon([oracleId])
  leaf = Poseidon([oracleId, oracleServiceType, repEff, cost, risk, latencyEst, cooldown])
  oraclePoolRoot = MerkleRoot(leaf, zero path, 8 levels)

It is called by zk_vos_pressure_test.py and should be placed in the same tools/ folder.
*/

const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : true;
      args[key] = val;
    }
  }
  return args;
}

function toStr(x) {
  if (typeof x === "bigint") return x.toString();
  if (typeof x === "number") return Math.trunc(x).toString();
  if (typeof x === "string") return x;
  return String(x);
}

async function main() {
  const args = parseArgs(process.argv);
  const jsonl = args.jsonl;
  const outDir = args["out-dir"];
  const levels = parseInt(args.levels || "8", 10);

  if (!jsonl || !outDir) {
    console.error("Usage: node zk_vos_enrich_full_inputs.js --jsonl base_inputs.jsonl --out-dir inputs --levels 8");
    process.exit(2);
  }

  let circomlibjs;
  try {
    circomlibjs = require("circomlibjs");
  } catch (e) {
    console.error("Cannot require('circomlibjs'). Run `npm install` in experiments/zk_vos first.");
    console.error(e.stack || e.toString());
    process.exit(1);
  }

  const poseidon = await circomlibjs.buildPoseidon();
  const F = poseidon.F;

  function pHash(arr) {
    return F.toObject(poseidon(arr.map((v) => BigInt(toStr(v))))).toString();
  }

  function merkleRoot(leaf, pathElements, pathIndices) {
    let cur = leaf.toString();
    for (let i = 0; i < levels; i++) {
      const sibling = (pathElements[i] || "0").toString();
      const idx = parseInt(pathIndices[i] || "0", 10);
      if (idx === 0) {
        cur = pHash([cur, sibling]);
      } else {
        cur = pHash([sibling, cur]);
      }
    }
    return cur;
  }

  fs.mkdirSync(outDir, { recursive: true });
  const lines = fs.readFileSync(jsonl, "utf8").split(/\r?\n/).filter(Boolean);

  for (let i = 0; i < lines.length; i++) {
    const b = JSON.parse(lines[i]);
    const pathElements = Array(levels).fill("0");
    const pathIndices = Array(levels).fill("0");

    const oracleId = toStr(b.oracleId ?? b.selectedOracleId ?? i);
    const oracleServiceType = toStr(b.oracleServiceType ?? b.requestServiceType ?? 0);
    const repEff = toStr(b.repEff ?? 6000);
    const cost = toStr(b.cost ?? 1000);
    const risk = toStr(b.risk ?? 600);
    const latencyEst = toStr(b.latencyEst ?? 6000);
    const cooldown = toStr(b.cooldown ?? 0);

    const selectedOracleHash = pHash([oracleId]);
    const leaf = pHash([oracleId, oracleServiceType, repEff, cost, risk, latencyEst, cooldown]);
    const oraclePoolRoot = merkleRoot(leaf, pathElements, pathIndices);

    const obj = {
      requestId: toStr(b.requestId ?? i),
      selectedOracleHash,
      oraclePoolRoot,
      reputationThreshold: toStr(b.reputationThreshold ?? 6000),
      costBudget: toStr(b.costBudget ?? 1000),
      riskBudget: toStr(b.riskBudget ?? 600),
      deadline: toStr(b.deadline ?? 6000),
      requestServiceType: toStr(b.requestServiceType ?? oracleServiceType),
      oracleId,
      oracleServiceType,
      repEff,
      cost,
      risk,
      latencyEst,
      cooldown,
      pathElements,
      pathIndices
    };

    const outPath = path.join(outDir, `input_${String(i).padStart(6, "0")}.json`);
    fs.writeFileSync(outPath, JSON.stringify(obj, null, 2));
  }

  console.log(`Generated ${lines.length} full-circuit inputs under ${outDir}`);
}

main().catch((e) => {
  console.error(e.stack || e.toString());
  process.exit(1);
});
