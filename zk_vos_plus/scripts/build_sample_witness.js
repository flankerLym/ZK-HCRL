const { buildPoseidon } = require("circomlibjs");

const LEVELS = 4;
const ZERO = 0n;

function toField(_poseidon, value) {
  // poseidonHash() already returns a native BigInt through F.toObject().
  // Passing that BigInt back into F.toObject() breaks on recent ffjavascript versions.
  return normalizeBigInt(value).toString();
}

function normalizeBigInt(v) {
  if (typeof v === "bigint") return v;
  if (typeof v === "number") return BigInt(v);
  if (typeof v === "string") return BigInt(v);
  throw new Error(`Unsupported bigint value: ${v}`);
}

function poseidonHash(poseidon, values) {
  return poseidon.F.toObject(poseidon(values.map(normalizeBigInt)));
}

function padLeaves(leaves, targetSize) {
  const out = leaves.slice();
  while (out.length < targetSize) out.push(ZERO);
  return out;
}

function buildTree(poseidon, leaves, levels = LEVELS) {
  const size = 1 << levels;
  let layer = padLeaves(leaves, size);
  const layers = [layer];
  for (let level = 0; level < levels; level++) {
    const next = [];
    for (let i = 0; i < layer.length; i += 2) {
      next.push(poseidonHash(poseidon, [layer[i], layer[i + 1]]));
    }
    layers.push(next);
    layer = next;
  }
  return { root: layers[levels][0], layers };
}

function getProof(tree, index, levels = LEVELS) {
  const pathElements = [];
  const pathIndices = [];
  let idx = index;
  for (let level = 0; level < levels; level++) {
    const sibling = idx ^ 1;
    pathElements.push(tree.layers[level][sibling]);
    pathIndices.push(idx & 1);
    idx = Math.floor(idx / 2);
  }
  return { pathElements, pathIndices };
}

function oracleLeaf(poseidon, oracle) {
  return poseidonHash(poseidon, [
    oracle.oracleId,
    oracle.addressHash,
    oracle.pubKeyHash,
    oracle.serviceType,
    oracle.cost,
    oracle.latency,
    oracle.capacity,
    oracle.stake
  ]);
}

function auditLeaf(poseidon, audit) {
  return poseidonHash(poseidon, [
    audit.oracleId,
    audit.reputation,
    audit.risk,
    audit.cooldown,
    audit.auditPassCount,
    audit.auditFailCount,
    audit.lastUpdateEpoch
  ]);
}

async function main() {
  const poseidon = await buildPoseidon();

  // Integer-scaled example. reputation/risk can use 0-10000 basis points.
  const oracles = [
    { oracleId: 1, addressHash: 10101, pubKeyHash: 20101, serviceType: 1, cost: 15, latency: 90, capacity: 100, stake: 1000 },
    { oracleId: 2, addressHash: 10102, pubKeyHash: 20102, serviceType: 1, cost: 17, latency: 110, capacity: 100, stake: 1000 },
    { oracleId: 3, addressHash: 10103, pubKeyHash: 20103, serviceType: 2, cost: 9, latency: 80, capacity: 80, stake: 800 },
    { oracleId: 4, addressHash: 10104, pubKeyHash: 20104, serviceType: 1, cost: 24, latency: 150, capacity: 120, stake: 1200 }
  ];

  const audits = [
    { oracleId: 1, reputation: 9300, risk: 120, cooldown: 0, auditPassCount: 97, auditFailCount: 3, lastUpdateEpoch: 4600 },
    { oracleId: 2, reputation: 9100, risk: 140, cooldown: 0, auditPassCount: 92, auditFailCount: 5, lastUpdateEpoch: 4600 },
    { oracleId: 3, reputation: 8000, risk: 300, cooldown: 0, auditPassCount: 70, auditFailCount: 10, lastUpdateEpoch: 4600 },
    { oracleId: 4, reputation: 5500, risk: 850, cooldown: 1, auditPassCount: 40, auditFailCount: 21, lastUpdateEpoch: 4600 }
  ];

  const poolLeaves = oracles.map(o => oracleLeaf(poseidon, o));
  const auditLeaves = audits.map(a => auditLeaf(poseidon, a));
  const poolTree = buildTree(poseidon, poolLeaves, LEVELS);
  const auditTree = buildTree(poseidon, auditLeaves, LEVELS);

  const primaryIndex = 0;
  const backupIndex = 1;
  const primary = oracles[primaryIndex];
  const backup = oracles[backupIndex];
  const primaryAudit = audits[primaryIndex];
  const backupAudit = audits[backupIndex];

  const mode = 2; // e.g., parallel_safe / backup-enabled mode in the paper's terminology.
  const backupEnabled = 1;
  const requestId = 1001;
  const requestType = 1;
  const costBudget = 40;
  const latencyBudget = 180;
  const riskBudget = 300;
  const reputationThreshold = 8500;

  const scheduleCommitment = poseidonHash(poseidon, [
    requestId,
    mode,
    backupEnabled,
    primary.oracleId,
    backup.oracleId,
    poolTree.root,
    auditTree.root
  ]);

  const pPoolProof = getProof(poolTree, primaryIndex, LEVELS);
  const bPoolProof = getProof(poolTree, backupIndex, LEVELS);
  const pAuditProof = getProof(auditTree, primaryIndex, LEVELS);
  const bAuditProof = getProof(auditTree, backupIndex, LEVELS);

  const forge = process.env.FORGE_SOURCE_MISMATCH === "1";
  const forgedPrimaryAuditOracleId = forge ? backupAudit.oracleId : primaryAudit.oracleId;

  const input = {
    requestId: String(requestId),
    requestType: String(requestType),
    costBudget: String(costBudget),
    latencyBudget: String(latencyBudget),
    riskBudget: String(riskBudget),
    reputationThreshold: String(reputationThreshold),
    oraclePoolRoot: toField(poseidon, poolTree.root),
    auditStateRoot: toField(poseidon, auditTree.root),
    scheduleCommitment: toField(poseidon, scheduleCommitment),

    mode: String(mode),
    backupEnabled: String(backupEnabled),

    primaryOracleId: String(primary.oracleId),
    primaryAddressHash: String(primary.addressHash),
    primaryPubKeyHash: String(primary.pubKeyHash),
    primaryServiceType: String(primary.serviceType),
    primaryCost: String(primary.cost),
    primaryLatency: String(primary.latency),
    primaryCapacity: String(primary.capacity),
    primaryStake: String(primary.stake),
    primaryPoolPathElements: pPoolProof.pathElements.map(x => toField(poseidon, x)),
    primaryPoolPathIndices: pPoolProof.pathIndices.map(String),

    primaryAuditOracleId: String(forgedPrimaryAuditOracleId),
    primaryReputation: String(primaryAudit.reputation),
    primaryRisk: String(primaryAudit.risk),
    primaryCooldown: String(primaryAudit.cooldown),
    primaryAuditPassCount: String(primaryAudit.auditPassCount),
    primaryAuditFailCount: String(primaryAudit.auditFailCount),
    primaryLastUpdateEpoch: String(primaryAudit.lastUpdateEpoch),
    primaryAuditPathElements: pAuditProof.pathElements.map(x => toField(poseidon, x)),
    primaryAuditPathIndices: pAuditProof.pathIndices.map(String),

    backupOracleId: String(backup.oracleId),
    backupAddressHash: String(backup.addressHash),
    backupPubKeyHash: String(backup.pubKeyHash),
    backupServiceType: String(backup.serviceType),
    backupCost: String(backup.cost),
    backupLatency: String(backup.latency),
    backupCapacity: String(backup.capacity),
    backupStake: String(backup.stake),
    backupPoolPathElements: bPoolProof.pathElements.map(x => toField(poseidon, x)),
    backupPoolPathIndices: bPoolProof.pathIndices.map(String),

    backupAuditOracleId: String(backupAudit.oracleId),
    backupReputation: String(backupAudit.reputation),
    backupRisk: String(backupAudit.risk),
    backupCooldown: String(backupAudit.cooldown),
    backupAuditPassCount: String(backupAudit.auditPassCount),
    backupAuditFailCount: String(backupAudit.auditFailCount),
    backupLastUpdateEpoch: String(backupAudit.lastUpdateEpoch),
    backupAuditPathElements: bAuditProof.pathElements.map(x => toField(poseidon, x)),
    backupAuditPathIndices: bAuditProof.pathIndices.map(String)
  };

  console.log(JSON.stringify(input, null, 2));
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
