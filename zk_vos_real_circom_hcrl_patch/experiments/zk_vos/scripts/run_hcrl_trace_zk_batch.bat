@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..

echo ============================================================
echo HCRL/Audit real-trace ZK-VOS batch experiment
echo Working dir: %CD%
echo ============================================================

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] node was not found in PATH.
  echo If you use nvm but PATH is broken, run this first in PowerShell:
  echo   $env:Path = "E:\Develop\node\nvm\v20.9.0;E:\keyan\DevelopTool\rust\.cargo\bin;$env:Path"
  exit /b 1
)

where circom >nul 2>nul
if errorlevel 1 (
  echo [ERROR] circom was not found in PATH.
  echo Install circom 2.x or add it to PATH, then rerun.
  exit /b 1
)

if not exist results mkdir results
if not exist build mkdir build
if not exist proof mkdir proof
if not exist inputs mkdir inputs

echo [0/9] Inspecting real HCRL trace data under data\ ...
node scripts\inspect_trace_data.js || exit /b 1

echo [1/9] Installing npm dependencies if needed...
if not exist node_modules call npm install --registry=https://registry.npmmirror.com || exit /b 1

echo [2/9] Building valid and mutated ZK inputs from real HCRL/Audit traces...
echo       validMode=relaxed keeps real selected HCRL oracle and relaxes public bounds when exported thresholds are too strict.
node scripts\build_real_trace_inputs.js --dataDir data --levels 8 --validMode relaxed --maxValid 1000 --maxPerMutation 1000 --out real_trace_inputs --singleOut inputs || exit /b 1

echo [3/9] Compiling Circom circuit...
circom circuits\zk_vos_full.circom --r1cs --wasm --sym -o build || exit /b 1

echo [4/9] Preparing Groth16 proving key if needed...
if not exist build\zk_vos_full_final.zkey (
  call npx snarkjs powersoftau new bn128 14 build\pot14_0000.ptau -v || exit /b 1
  call npx snarkjs powersoftau contribute build\pot14_0000.ptau build\pot14_0001.ptau --name="zk-vos-real-trace" -v -e="zk-vos-real-trace-entropy" || exit /b 1
  call npx snarkjs powersoftau prepare phase2 build\pot14_0001.ptau build\pot14_final.ptau -v || exit /b 1
  call npx snarkjs groth16 setup build\zk_vos_full.r1cs build\pot14_final.ptau build\zk_vos_full_0000.zkey || exit /b 1
  call npx snarkjs zkey contribute build\zk_vos_full_0000.zkey build\zk_vos_full_final.zkey --name="zk-vos-real-trace-zkey" -v -e="zk-vos-real-trace-zkey-entropy" || exit /b 1
)
call npx snarkjs zkey export verificationkey build\zk_vos_full_final.zkey build\verification_key.json || exit /b 1

echo [5/9] Batch witness validation for valid and invalid real-trace inputs...
node scripts\check_real_trace_batch.js --in real_trace_inputs --maxValid 1000 --maxInvalidPerCase 1000 || exit /b 1

echo [6/9] Timing Groth16 proof generation on real HCRL valid schedules...
node scripts\time_real_trace_proofs.js --in real_trace_inputs --maxProofs 20 || exit /b 1

echo [7/9] Exporting Solidity verifier...
call npx snarkjs zkey export solidityverifier build\zk_vos_full_final.zkey contracts\Verifier.sol || exit /b 1

echo [8/9] Running Hardhat registry test with real proof calldata and gas report...
call npx hardhat test test\real_proof_registry.test.js > results\hcrl_trace_real_registry_gas.txt 2>&1
if errorlevel 1 (
  type results\hcrl_trace_real_registry_gas.txt
  exit /b 1
)
type results\hcrl_trace_real_registry_gas.txt

echo [9/9] Summarizing real-trace ZK-VOS results...
node scripts\summarize_hcrl_trace_zk_results.js || exit /b 1

echo.
echo [DONE] HCRL/Audit real-trace ZK-VOS batch experiment completed.
echo Key outputs:
echo   results\hcrl_trace_batch_summary.csv
echo   results\hcrl_trace_batch_validation.csv
echo   results\hcrl_trace_proof_timing.csv
echo   results\hcrl_trace_real_registry_gas.txt
echo   results\hcrl_trace_zk_final_summary.json
