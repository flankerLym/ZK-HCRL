param(
  [string]$TraceCsv = "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/data/trace_hcrl_zk_schedule_trace.csv",
  [string]$Wasm = "zk_vos_real_circom_hcrl_patch/circuits/zk_vos_js/zk_vos.wasm",
  [string]$Zkey = "zk_vos_real_circom_hcrl_patch/circuits/zk_vos_final.zkey",
  [string]$Vkey = "zk_vos_real_circom_hcrl_patch/circuits/verification_key.json",
  [string]$Sizes = "100 500 1000 5000",
  [int]$SingleVerifyGas = 0,
  [string]$Snarkjs = "snarkjs"
)

$ErrorActionPreference = "Stop"

if ($SingleVerifyGas -gt 0) {
  python "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py" stress `
    --snarkjs $Snarkjs `
    --trace-csv $TraceCsv `
    --wasm $Wasm `
    --zkey $Zkey `
    --vkey $Vkey `
    --sizes $Sizes `
    --single-verify-gas $SingleVerifyGas `
    --repeat-trace-rows
} else {
  python "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py" stress `
    --snarkjs $Snarkjs `
    --trace-csv $TraceCsv `
    --wasm $Wasm `
    --zkey $Zkey `
    --vkey $Vkey `
    --sizes $Sizes `
    --repeat-trace-rows
}
