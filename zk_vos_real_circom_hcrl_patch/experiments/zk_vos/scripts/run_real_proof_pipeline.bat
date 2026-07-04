@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..

where circom >nul 2>nul
if errorlevel 1 (
  echo [ERROR] circom was not found in PATH.
  echo Install circom 2.x first, then rerun this script.
  echo See README_ZK_VOS_REAL_PROOF_AND_HCRL.md for options.
  exit /b 1
)

if not exist build mkdir build
if not exist proof mkdir proof
if not exist results mkdir results
if not exist inputs mkdir inputs

set START_ALL=%TIME%
echo [1/10] Installing npm dependencies if needed...
if not exist node_modules call npm install || exit /b 1

echo [2/10] Generating ZK inputs from oracle pool and HCRL-like schedule CSV...
node scripts\generate_inputs.js --row 0 --levels 8 || exit /b 1

echo [3/10] Compiling Circom circuit...
circom circuits\zk_vos_full.circom --r1cs --wasm --sym -o build || exit /b 1

echo [4/10] Running Groth16 demo trusted setup...
call npx snarkjs powersoftau new bn128 14 build\pot14_0000.ptau -v || exit /b 1
call npx snarkjs powersoftau contribute build\pot14_0000.ptau build\pot14_0001.ptau --name="zk-vos-demo" -v -e="zk-vos-demo-entropy" || exit /b 1
call npx snarkjs powersoftau prepare phase2 build\pot14_0001.ptau build\pot14_final.ptau -v || exit /b 1
call npx snarkjs groth16 setup build\zk_vos_full.r1cs build\pot14_final.ptau build\zk_vos_full_0000.zkey || exit /b 1
call npx snarkjs zkey contribute build\zk_vos_full_0000.zkey build\zk_vos_full_final.zkey --name="zk-vos-zkey" -v -e="zk-vos-zkey-entropy" || exit /b 1
call npx snarkjs zkey export verificationkey build\zk_vos_full_final.zkey build\verification_key.json || exit /b 1

echo [5/10] Generating valid witness and proof...
node build\zk_vos_full_js\generate_witness.js build\zk_vos_full_js\zk_vos_full.wasm inputs\valid_schedule.json build\valid.wtns || exit /b 1
call npx snarkjs groth16 prove build\zk_vos_full_final.zkey build\valid.wtns proof\valid_proof.json proof\valid_public.json || exit /b 1

echo [6/10] Verifying proof off-chain...
call npx snarkjs groth16 verify build\verification_key.json proof\valid_public.json proof\valid_proof.json > results\offchain_verify_valid.txt || exit /b 1

echo [7/10] Checking invalid schedules are rejected by circuit constraints...
node scripts\check_invalid_witnesses.js || exit /b 1

echo [8/10] Exporting Solidity Groth16 verifier and calldata...
call npx snarkjs zkey export solidityverifier build\zk_vos_full_final.zkey contracts\Verifier.sol || exit /b 1
call npx snarkjs zkey export soliditycalldata proof\valid_public.json proof\valid_proof.json > proof\valid_calldata_raw.txt || exit /b 1
node scripts\format_snarkjs_calldata.js proof\valid_calldata_raw.txt proof\valid_calldata.json || exit /b 1

echo [9/10] Running Hardhat real proof verifier test with gas reporter...
call npx hardhat test test\real_proof_registry.test.js || exit /b 1

echo [10/10] Collecting proof artifacts metrics...
node scripts\collect_real_proof_metrics.js || exit /b 1

echo.
echo [DONE] Real Circom proof + Solidity Groth16 verifier pipeline completed.
echo Results: experiments\zk_vos\results\
echo Proofs:  experiments\zk_vos\proof\
