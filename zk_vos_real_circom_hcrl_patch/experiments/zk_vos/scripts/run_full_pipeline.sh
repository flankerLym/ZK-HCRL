#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p build proof results contracts

echo "[1/8] Generating valid/invalid ZK inputs..."
node scripts/generate_inputs.js

echo "[2/8] Compiling Circom circuit..."
circom circuits/zk_vos_full.circom --r1cs --wasm --sym -o build

echo "[3/8] Running Groth16 demo setup..."
# Demo-only Powers of Tau. For a production system, use a real ceremony.
npx snarkjs powersoftau new bn128 14 build/pot14_0000.ptau -v
npx snarkjs powersoftau contribute build/pot14_0000.ptau build/pot14_0001.ptau --name="zk-vos-demo" -v -e="zk-vos-demo-entropy"
npx snarkjs powersoftau prepare phase2 build/pot14_0001.ptau build/pot14_final.ptau -v
npx snarkjs groth16 setup build/zk_vos_full.r1cs build/pot14_final.ptau build/zk_vos_full_0000.zkey
npx snarkjs zkey contribute build/zk_vos_full_0000.zkey build/zk_vos_full_final.zkey --name="zk-vos-zkey" -v -e="zk-vos-zkey-entropy"
npx snarkjs zkey export verificationkey build/zk_vos_full_final.zkey build/verification_key.json

echo "[4/8] Generating witness and proof for valid schedule..."
node build/zk_vos_full_js/generate_witness.js build/zk_vos_full_js/zk_vos_full.wasm inputs/valid_schedule.json build/valid.wtns
npx snarkjs groth16 prove build/zk_vos_full_final.zkey build/valid.wtns proof/valid_proof.json proof/valid_public.json

echo "[5/8] Verifying proof off-chain..."
npx snarkjs groth16 verify build/verification_key.json proof/valid_public.json proof/valid_proof.json | tee results/offchain_verify_valid.txt

echo "[6/8] Exporting Solidity verifier..."
npx snarkjs zkey export solidityverifier build/zk_vos_full_final.zkey contracts/Verifier.sol

echo "[7/8] Exporting calldata..."
npx snarkjs zkey export soliditycalldata proof/valid_public.json proof/valid_proof.json > proof/valid_calldata_raw.txt
node scripts/format_snarkjs_calldata.js proof/valid_calldata_raw.txt proof/valid_calldata.json

echo "[8/8] Running Hardhat tests..."
npx hardhat test

echo "Done. Results are in experiments/zk_vos/results and experiments/zk_vos/proof."
