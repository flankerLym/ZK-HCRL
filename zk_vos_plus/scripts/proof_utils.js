function groth16ToSolidityCalldata(proof) {
  return {
    a: [proof.pi_a[0], proof.pi_a[1]],
    // snarkjs exports B in the order expected by Solidity verifier after swapping the inner pairs.
    b: [
      [proof.pi_b[0][1], proof.pi_b[0][0]],
      [proof.pi_b[1][1], proof.pi_b[1][0]]
    ],
    c: [proof.pi_c[0], proof.pi_c[1]]
  };
}

function normalizePublicSignals(publicSignals) {
  if (publicSignals.length !== 9) {
    throw new Error(`Expected 9 public signals, got ${publicSignals.length}`);
  }
  return publicSignals.map(x => BigInt(x).toString());
}

module.exports = { groth16ToSolidityCalldata, normalizePublicSignals };
