#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v circom >/dev/null 2>&1; then
  echo "[ERROR] circom was not found in PATH. Install circom 2.x first."
  echo "See README_ZK_VOS_REAL_PROOF_AND_HCRL.md for options."
  exit 1
fi

mkdir -p build proof results inputs

if [ ! -d node_modules ]; then
  npm install
fi

echo "[1/10] Generating ZK inputs from oracle pool and HCRL-like schedule CSV..."
node scripts/generate_inputs.js --row 0 --levels 8

echo "[2/10] Compiling Circom circuit..."
circom circuits/zk_vos_full.circom --r1cs --wasm --sym -o build

echo "[3/10] Running Groth16 demo trusted setup..."
npx snarkjs powersoftau new bn128 14 build/pot14_0000.ptau -v
npx snarkjs powersoftau contribute build/pot14_0000.ptau build/pot14_0001.ptau --name="zk-vos-demo" -v -e="zk-vos-demo-entropy"
npx snarkjs powersoftau prepare phase2 build/pot14_0001.ptau build/pot14_final.ptau -v
npx snarkjs groth16 setup build/zk_vos_full.r1cs build/pot14_final.ptau build/zk_vos_full_0000.zkey
npx snarkjs zkey contribute build/zk_vos_full_0000.zkey build/zk_vos_full_final.zkey --name="zk-vos-zkey" -v -e="zk-vos-zkey-entropy"
npx snarkjs zkey export verificationkey build/zk_vos_full_final.zkey build/verification_key.json

echo "[4/10] Generating valid witness and proof..."
node build/zk_vos_full_js/generate_witness.js build/zk_vos_full_js/zk_vos_full.wasm inputs/valid_schedule.json build/valid.wtns
npx snarkjs groth16 prove build/zk_vos_full_final.zkey build/valid.wtns proof/valid_proof.json proof/valid_public.json

echo "[5/10] Verifying proof off-chain..."
npx snarkjs groth16 verify build/verification_key.json proof/valid_public.json proof/valid_proof.json | tee results/offchain_verify_valid.txt

echo "[6/10] Checking invalid schedules are rejected by circuit constraints..."
node scripts/check_invalid_witnesses.js

echo "[7/10] Exporting Solidity Groth16 verifier and calldata..."
npx snarkjs zkey export solidityverifier build/zk_vos_full_final.zkey contracts/Verifier.sol
npx snarkjs zkey export soliditycalldata proof/valid_public.json proof/valid_proof.json > proof/valid_calldata_raw.txt
node scripts/format_snarkjs_calldata.js proof/valid_calldata_raw.txt proof/valid_calldata.json

echo "[8/10] Running Hardhat real proof verifier test with gas reporter..."
npx hardhat test test/real_proof_registry.test.js

echo "[9/10] Collecting proof artifacts metrics..."
node scripts/collect_real_proof_metrics.js

echo "[DONE] Real Circom proof + Solidity Groth16 verifier pipeline completed."
