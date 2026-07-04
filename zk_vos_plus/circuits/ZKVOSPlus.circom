pragma circom 2.1.6;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";

// Merkle membership verifier using Poseidon(2).
// pathIndices[i] = 0 means current hash is the left child; 1 means right child.
template MerkleInclusion(levels) {
    signal input leaf;
    signal input root;
    signal input pathElements[levels];
    signal input pathIndices[levels];
    signal output isValid;

    signal cur[levels + 1];
    signal left[levels];
    signal right[levels];
    component h[levels];
    component eq = IsEqual();

    cur[0] <== leaf;
    for (var i = 0; i < levels; i++) {
        // Boolean constraint for path direction.
        pathIndices[i] * (pathIndices[i] - 1) === 0;

        // Use one multiplication per mux to keep every constraint quadratic:
        // left  = cur + bit * (sibling - cur)
        // right = sibling + bit * (cur - sibling)
        left[i] <== cur[i] + pathIndices[i] * (pathElements[i] - cur[i]);
        right[i] <== pathElements[i] + pathIndices[i] * (cur[i] - pathElements[i]);

        h[i] = Poseidon(2);
        h[i].inputs[0] <== left[i];
        h[i].inputs[1] <== right[i];
        cur[i + 1] <== h[i].out;
    }

    eq.in[0] <== cur[levels];
    eq.in[1] <== root;
    isValid <== eq.out;
}

template OraclePoolLeaf() {
    signal input oracleId;
    signal input oracleAddressHash;
    signal input oraclePubKeyHash;
    signal input serviceType;
    signal input cost;
    signal input latency;
    signal input capacity;
    signal input stake;
    signal output leaf;

    component h = Poseidon(8);
    h.inputs[0] <== oracleId;
    h.inputs[1] <== oracleAddressHash;
    h.inputs[2] <== oraclePubKeyHash;
    h.inputs[3] <== serviceType;
    h.inputs[4] <== cost;
    h.inputs[5] <== latency;
    h.inputs[6] <== capacity;
    h.inputs[7] <== stake;
    leaf <== h.out;
}

template AuditStateLeaf() {
    signal input oracleId;
    signal input reputation;
    signal input risk;
    signal input cooldown;
    signal input auditPassCount;
    signal input auditFailCount;
    signal input lastUpdateEpoch;
    signal output leaf;

    component h = Poseidon(7);
    h.inputs[0] <== oracleId;
    h.inputs[1] <== reputation;
    h.inputs[2] <== risk;
    h.inputs[3] <== cooldown;
    h.inputs[4] <== auditPassCount;
    h.inputs[5] <== auditFailCount;
    h.inputs[6] <== lastUpdateEpoch;
    leaf <== h.out;
}

// Two-layer ZK-VOS+:
// Layer 1: compliance validity: cost/latency/risk/reputation/cooldown/service constraints.
// Layer 2: identity-state consistency: selected oracle IDs are bound to oraclePoolRoot and auditStateRoot.
template ZKVOSPlus(levels, nBits) {
    // Public inputs.
    signal input requestId;
    signal input requestType;
    signal input costBudget;
    signal input latencyBudget;
    signal input riskBudget;
    signal input reputationThreshold;
    signal input oraclePoolRoot;
    signal input auditStateRoot;
    signal input scheduleCommitment;

    // Private scheduling witnesses.
    signal input mode;
    signal input backupEnabled;

    // Primary oracle registered attributes.
    signal input primaryOracleId;
    signal input primaryAddressHash;
    signal input primaryPubKeyHash;
    signal input primaryServiceType;
    signal input primaryCost;
    signal input primaryLatency;
    signal input primaryCapacity;
    signal input primaryStake;
    signal input primaryPoolPathElements[levels];
    signal input primaryPoolPathIndices[levels];

    // Primary audit state.
    signal input primaryAuditOracleId;
    signal input primaryReputation;
    signal input primaryRisk;
    signal input primaryCooldown;
    signal input primaryAuditPassCount;
    signal input primaryAuditFailCount;
    signal input primaryLastUpdateEpoch;
    signal input primaryAuditPathElements[levels];
    signal input primaryAuditPathIndices[levels];

    // Backup oracle registered attributes.
    signal input backupOracleId;
    signal input backupAddressHash;
    signal input backupPubKeyHash;
    signal input backupServiceType;
    signal input backupCost;
    signal input backupLatency;
    signal input backupCapacity;
    signal input backupStake;
    signal input backupPoolPathElements[levels];
    signal input backupPoolPathIndices[levels];

    // Backup audit state.
    signal input backupAuditOracleId;
    signal input backupReputation;
    signal input backupRisk;
    signal input backupCooldown;
    signal input backupAuditPassCount;
    signal input backupAuditFailCount;
    signal input backupLastUpdateEpoch;
    signal input backupAuditPathElements[levels];
    signal input backupAuditPathIndices[levels];

    // backupEnabled must be boolean.
    backupEnabled * (backupEnabled - 1) === 0;

    // Merkle leaves and membership checks.
    component pPoolLeaf = OraclePoolLeaf();
    pPoolLeaf.oracleId <== primaryOracleId;
    pPoolLeaf.oracleAddressHash <== primaryAddressHash;
    pPoolLeaf.oraclePubKeyHash <== primaryPubKeyHash;
    pPoolLeaf.serviceType <== primaryServiceType;
    pPoolLeaf.cost <== primaryCost;
    pPoolLeaf.latency <== primaryLatency;
    pPoolLeaf.capacity <== primaryCapacity;
    pPoolLeaf.stake <== primaryStake;

    component pPoolProof = MerkleInclusion(levels);
    pPoolProof.leaf <== pPoolLeaf.leaf;
    pPoolProof.root <== oraclePoolRoot;
    for (var i = 0; i < levels; i++) {
        pPoolProof.pathElements[i] <== primaryPoolPathElements[i];
        pPoolProof.pathIndices[i] <== primaryPoolPathIndices[i];
    }
    pPoolProof.isValid === 1;

    component pAuditLeaf = AuditStateLeaf();
    pAuditLeaf.oracleId <== primaryAuditOracleId;
    pAuditLeaf.reputation <== primaryReputation;
    pAuditLeaf.risk <== primaryRisk;
    pAuditLeaf.cooldown <== primaryCooldown;
    pAuditLeaf.auditPassCount <== primaryAuditPassCount;
    pAuditLeaf.auditFailCount <== primaryAuditFailCount;
    pAuditLeaf.lastUpdateEpoch <== primaryLastUpdateEpoch;

    component pAuditProof = MerkleInclusion(levels);
    pAuditProof.leaf <== pAuditLeaf.leaf;
    pAuditProof.root <== auditStateRoot;
    for (var j = 0; j < levels; j++) {
        pAuditProof.pathElements[j] <== primaryAuditPathElements[j];
        pAuditProof.pathIndices[j] <== primaryAuditPathIndices[j];
    }
    pAuditProof.isValid === 1;

    // Identity consistency: the selected primary oracle must be the same entity in the pool and audit-state trees.
    primaryOracleId === primaryAuditOracleId;

    component bPoolLeaf = OraclePoolLeaf();
    bPoolLeaf.oracleId <== backupOracleId;
    bPoolLeaf.oracleAddressHash <== backupAddressHash;
    bPoolLeaf.oraclePubKeyHash <== backupPubKeyHash;
    bPoolLeaf.serviceType <== backupServiceType;
    bPoolLeaf.cost <== backupCost;
    bPoolLeaf.latency <== backupLatency;
    bPoolLeaf.capacity <== backupCapacity;
    bPoolLeaf.stake <== backupStake;

    component bPoolProof = MerkleInclusion(levels);
    bPoolProof.leaf <== bPoolLeaf.leaf;
    bPoolProof.root <== oraclePoolRoot;
    for (var k = 0; k < levels; k++) {
        bPoolProof.pathElements[k] <== backupPoolPathElements[k];
        bPoolProof.pathIndices[k] <== backupPoolPathIndices[k];
    }
    // If backup is enabled, its pool membership must verify.
    backupEnabled * (bPoolProof.isValid - 1) === 0;

    component bAuditLeaf = AuditStateLeaf();
    bAuditLeaf.oracleId <== backupAuditOracleId;
    bAuditLeaf.reputation <== backupReputation;
    bAuditLeaf.risk <== backupRisk;
    bAuditLeaf.cooldown <== backupCooldown;
    bAuditLeaf.auditPassCount <== backupAuditPassCount;
    bAuditLeaf.auditFailCount <== backupAuditFailCount;
    bAuditLeaf.lastUpdateEpoch <== backupLastUpdateEpoch;

    component bAuditProof = MerkleInclusion(levels);
    bAuditProof.leaf <== bAuditLeaf.leaf;
    bAuditProof.root <== auditStateRoot;
    for (var l = 0; l < levels; l++) {
        bAuditProof.pathElements[l] <== backupAuditPathElements[l];
        bAuditProof.pathIndices[l] <== backupAuditPathIndices[l];
    }
    // If backup is enabled, its audit-state membership and ID consistency must verify.
    backupEnabled * (bAuditProof.isValid - 1) === 0;
    backupEnabled * (backupOracleId - backupAuditOracleId) === 0;

    // If backup is enabled, backup must not equal primary.
    component neq = IsEqual();
    neq.in[0] <== primaryOracleId;
    neq.in[1] <== backupOracleId;
    backupEnabled * neq.out === 0;

    // Service compatibility.
    component pSvcEq = IsEqual();
    pSvcEq.in[0] <== primaryServiceType;
    pSvcEq.in[1] <== requestType;
    pSvcEq.out === 1;

    component bSvcEq = IsEqual();
    bSvcEq.in[0] <== backupServiceType;
    bSvcEq.in[1] <== requestType;
    backupEnabled * (bSvcEq.out - 1) === 0;

    // Compliance constraints.
    component pRiskOk = LessEqThan(nBits);
    pRiskOk.in[0] <== primaryRisk;
    pRiskOk.in[1] <== riskBudget;
    pRiskOk.out === 1;

    component pRepOk = LessEqThan(nBits);
    pRepOk.in[0] <== reputationThreshold;
    pRepOk.in[1] <== primaryReputation;
    pRepOk.out === 1;

    primaryCooldown === 0;

    component bRiskOk = LessEqThan(nBits);
    bRiskOk.in[0] <== backupRisk;
    bRiskOk.in[1] <== riskBudget;
    backupEnabled * (bRiskOk.out - 1) === 0;

    component bRepOk = LessEqThan(nBits);
    bRepOk.in[0] <== reputationThreshold;
    bRepOk.in[1] <== backupReputation;
    backupEnabled * (bRepOk.out - 1) === 0;

    backupEnabled * backupCooldown === 0;

    signal totalCost;
    signal combinedRisk;
    totalCost <== primaryCost + backupEnabled * backupCost;
    combinedRisk <== primaryRisk + backupEnabled * backupRisk;

    component costOk = LessEqThan(nBits);
    costOk.in[0] <== totalCost;
    costOk.in[1] <== costBudget;
    costOk.out === 1;

    // Conservative risk bound for primary-backup schedules.
    component riskOk = LessEqThan(nBits);
    riskOk.in[0] <== combinedRisk;
    riskOk.in[1] <== riskBudget;
    riskOk.out === 1;

    // Latency feasibility: every selected oracle must be individually within the deadline.
    component pLatOk = LessEqThan(nBits);
    pLatOk.in[0] <== primaryLatency;
    pLatOk.in[1] <== latencyBudget;
    pLatOk.out === 1;

    component bLatOk = LessEqThan(nBits);
    bLatOk.in[0] <== backupLatency;
    bLatOk.in[1] <== latencyBudget;
    backupEnabled * (bLatOk.out - 1) === 0;

    // Bind the submitted schedule commitment to request, mode, selected identities and committed states.
    component schHash = Poseidon(7);
    schHash.inputs[0] <== requestId;
    schHash.inputs[1] <== mode;
    schHash.inputs[2] <== backupEnabled;
    schHash.inputs[3] <== primaryOracleId;
    schHash.inputs[4] <== backupOracleId;
    schHash.inputs[5] <== oraclePoolRoot;
    schHash.inputs[6] <== auditStateRoot;
    schHash.out === scheduleCommitment;
}

component main { public [requestId, requestType, costBudget, latencyBudget, riskBudget, reputationThreshold, oraclePoolRoot, auditStateRoot, scheduleCommitment] } = ZKVOSPlus(4, 32);
