#!/usr/bin/env bash
set -euo pipefail

TRACE_CSV=${TRACE_CSV:-"zk_vos_real_circom_hcrl_patch/experiments/zk_vos/data/trace_hcrl_zk_schedule_trace.csv"}
WASM=${WASM:-"zk_vos_real_circom_hcrl_patch/circuits/zk_vos_js/zk_vos.wasm"}
ZKEY=${ZKEY:-"zk_vos_real_circom_hcrl_patch/circuits/zk_vos_final.zkey"}
VKEY=${VKEY:-"zk_vos_real_circom_hcrl_patch/circuits/verification_key.json"}
SIZES=${SIZES:-"100 500 1000 5000"}
SNARKJS=${SNARKJS:-"snarkjs"}
SINGLE_VERIFY_GAS=${SINGLE_VERIFY_GAS:-""}

cmd=(python "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py" stress
  --snarkjs "$SNARKJS"
  --trace-csv "$TRACE_CSV"
  --wasm "$WASM"
  --zkey "$ZKEY"
  --vkey "$VKEY"
  --sizes "$SIZES"
  --repeat-trace-rows)

if [[ -n "$SINGLE_VERIFY_GAS" ]]; then
  cmd+=(--single-verify-gas "$SINGLE_VERIFY_GAS")
fi

"${cmd[@]}"
