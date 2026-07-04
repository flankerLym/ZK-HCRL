pragma circom 2.1.6;

include "../node_modules/circomlib/circuits/comparators.circom";
include "../node_modules/circomlib/circuits/poseidon.circom";

// A simple Poseidon Merkle root checker.
// The leaf commits to private oracle state:
// leaf = Poseidon(oracleId, oracleServiceType, repEff, cost, risk, latencyEst, cooldown)
// The root is public.
template MerkleRoot(levels) {
    signal input leaf;
    signal input pathElements[levels];
    signal input pathIndices[levels];
    signal output root;

    signal cur[levels + 1];
    cur[0] <== leaf;

    component hashers[levels];
    signal left[levels];
    signal right[levels];
    signal isRight[levels];
    signal isLeft[levels];

    for (var i = 0; i < levels; i++) {
        // Boolean constraint for path index.
        pathIndices[i] * (pathIndices[i] - 1) === 0;
        isRight[i] <== pathIndices[i];
        isLeft[i] <== 1 - pathIndices[i];

        // If path index is 0, current node is left and sibling is right.
        // If path index is 1, sibling is left and current node is right.
        left[i] <== pathElements[i] + isLeft[i] * (cur[i] - pathElements[i]);
        right[i] <== cur[i] + isLeft[i] * (pathElements[i] - cur[i]);

        hashers[i] = Poseidon(2);
        hashers[i].inputs[0] <== left[i];
        hashers[i].inputs[1] <== right[i];
        cur[i + 1] <== hashers[i].out;
    }

    root <== cur[levels];
}

template ZKVOSFull(levels) {
    // Public inputs
    signal input requestId;
    signal input selectedOracleHash;
    signal input oraclePoolRoot;
    signal input reputationThreshold;
    signal input costBudget;
    signal input riskBudget;
    signal input deadline;
    signal input requestServiceType;

    // Private witness inputs
    signal input oracleId;
    signal input oracleServiceType;
    signal input repEff;
    signal input cost;
    signal input risk;
    signal input latencyEst;
    signal input cooldown;
    signal input pathElements[levels];
    signal input pathIndices[levels];

    // Bind selected oracle id to a public hash.
    component oracleIdHasher = Poseidon(1);
    oracleIdHasher.inputs[0] <== oracleId;
    oracleIdHasher.out === selectedOracleHash;

    // Service matching.
    oracleServiceType === requestServiceType;

    // Reputation: repEff >= reputationThreshold.
    component repOk = LessEqThan(32);
    repOk.in[0] <== reputationThreshold;
    repOk.in[1] <== repEff;
    repOk.out === 1;

    // Cost: cost <= costBudget.
    component costOk = LessEqThan(32);
    costOk.in[0] <== cost;
    costOk.in[1] <== costBudget;
    costOk.out === 1;

    // Risk: risk <= riskBudget.
    component riskOk = LessEqThan(32);
    riskOk.in[0] <== risk;
    riskOk.in[1] <== riskBudget;
    riskOk.out === 1;

    // Latency: latencyEst <= deadline.
    component latencyOk = LessEqThan(32);
    latencyOk.in[0] <== latencyEst;
    latencyOk.in[1] <== deadline;
    latencyOk.out === 1;

    // Cooldown must be zero.
    cooldown === 0;

    // Commit all private oracle state into a Merkle leaf.
    component leafHasher = Poseidon(7);
    leafHasher.inputs[0] <== oracleId;
    leafHasher.inputs[1] <== oracleServiceType;
    leafHasher.inputs[2] <== repEff;
    leafHasher.inputs[3] <== cost;
    leafHasher.inputs[4] <== risk;
    leafHasher.inputs[5] <== latencyEst;
    leafHasher.inputs[6] <== cooldown;

    component merkle = MerkleRoot(levels);
    merkle.leaf <== leafHasher.out;
    for (var i = 0; i < levels; i++) {
        merkle.pathElements[i] <== pathElements[i];
        merkle.pathIndices[i] <== pathIndices[i];
    }
    merkle.root === oraclePoolRoot;
}

component main { public [
    requestId,
    selectedOracleHash,
    oraclePoolRoot,
    reputationThreshold,
    costBudget,
    riskBudget,
    deadline,
    requestServiceType
] } = ZKVOSFull(8);
