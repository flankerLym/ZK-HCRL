@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..

if not exist build mkdir build
if not exist proof mkdir proof
if not exist results mkdir results
if not exist contracts mkdir contracts

echo [1/8] Generating valid/invalid ZK inputs...
node scripts\generate_inputs.js || exit /b 1

echo [2/8] Compiling Circom circuit...
circom circuits\zk_vos_full.circom --r1cs --wasm --sym -o build || exit /b 1

echo [3/8] Running Groth16 demo setup...
call npx snarkjs powersoftau new bn128 14 build\pot14_0000.ptau -v || exit /b 1
call npx snarkjs powersoftau contribute build\pot14_0000.ptau build\pot14_0001.ptau --name="zk-vos-demo" -v -e="zk-vos-demo-entropy" || exit /b 1
call npx snarkjs powersoftau prepare phase2 build\pot14_0001.ptau build\pot14_final.ptau -v || exit /b 1
call npx snarkjs groth16 setup build\zk_vos_full.r1cs build\pot14_final.ptau build\zk_vos_full_0000.zkey || exit /b 1
call npx snarkjs zkey contribute build\zk_vos_full_0000.zkey build\zk_vos_full_final.zkey --name="zk-vos-zkey" -v -e="zk-vos-zkey-entropy" || exit /b 1
call npx snarkjs zkey export verificationkey build\zk_vos_full_final.zkey build\verification_key.json || exit /b 1

echo [4/8] Generating witness and proof for valid schedule...
node build\zk_vos_full_js\generate_witness.js build\zk_vos_full_js\zk_vos_full.wasm inputs\valid_schedule.json build\valid.wtns || exit /b 1
call npx snarkjs groth16 prove build\zk_vos_full_final.zkey build\valid.wtns proof\valid_proof.json proof\valid_public.json || exit /b 1

echo [5/8] Verifying proof off-chain...
call npx snarkjs groth16 verify build\verification_key.json proof\valid_public.json proof\valid_proof.json > results\offchain_verify_valid.txt || exit /b 1

echo [6/8] Exporting Solidity verifier...
call npx snarkjs zkey export solidityverifier build\zk_vos_full_final.zkey contracts\Verifier.sol || exit /b 1

echo [7/8] Exporting calldata...
call npx snarkjs zkey export soliditycalldata proof\valid_public.json proof\valid_proof.json > proof\valid_calldata_raw.txt || exit /b 1
node scripts\format_snarkjs_calldata.js proof\valid_calldata_raw.txt proof\valid_calldata.json || exit /b 1

echo [8/8] Running Hardhat tests...
call npx hardhat test || exit /b 1

echo Done. Results are in experiments\zk_vos\results and experiments\zk_vos\proof.
pause
