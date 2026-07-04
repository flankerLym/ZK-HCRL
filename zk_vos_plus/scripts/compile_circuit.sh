#!/usr/bin/env bash
set -euo pipefail

mkdir -p testnet_artifacts
circom circuits/ZKVOSPlus.circom --r1cs --wasm --sym -l node_modules -o testnet_artifacts
snarkjs r1cs info testnet_artifacts/ZKVOSPlus.r1cs

# Small demo powers of tau. For paper experiments, run a fresh ceremony or clearly describe this setup.
snarkjs powersoftau new bn128 14 testnet_artifacts/pot14_0000.ptau -v
snarkjs powersoftau contribute testnet_artifacts/pot14_0000.ptau testnet_artifacts/pot14_0001.ptau --name="ZK-VOS+ demo contribution" -v -e="zk-vos-plus"
snarkjs powersoftau prepare phase2 testnet_artifacts/pot14_0001.ptau testnet_artifacts/pot14_final.ptau -v
snarkjs groth16 setup testnet_artifacts/ZKVOSPlus.r1cs testnet_artifacts/pot14_final.ptau testnet_artifacts/ZKVOSPlus_0000.zkey
snarkjs zkey contribute testnet_artifacts/ZKVOSPlus_0000.zkey testnet_artifacts/ZKVOSPlus_final.zkey --name="ZK-VOS+ final" -v -e="zk-vos-plus-final"
snarkjs zkey export verificationkey testnet_artifacts/ZKVOSPlus_final.zkey testnet_artifacts/verification_key.json
snarkjs zkey export solidityverifier testnet_artifacts/ZKVOSPlus_final.zkey contracts/ZKVOSPlusVerifier.sol

echo "Circuit artifacts written to testnet_artifacts/ and contracts/ZKVOSPlusVerifier.sol"
