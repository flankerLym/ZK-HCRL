#!/usr/bin/env bash
set -euo pipefail

# ZK-VOS batch verification / scalability experiment.
# Run this from the repository root or from `TCO-DRL_with baseline/`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRACE_CSV="${TRACE_CSV:-}"
VKEY="${VKEY:-}"
WASM="${WASM:-}"
ZKEY="${ZKEY:-}"
PROOF_DIR="${PROOF_DIR:-output/zk_vos_scalability/latest_proofs}"
PUBLIC_DIR="${PUBLIC_DIR:-output/zk_vos_scalability/latest_public}"
OUT_DIR="${OUT_DIR:-output/zk_vos_scalability}"
SIZES="${SIZES:-100 500 1000 2000 5000}"
MODE="${MODE:-verify}"

if [[ "$MODE" == "prepare-inputs" || "$MODE" == "prove" || "$MODE" == "full" ]]; then
  if [[ -z "$TRACE_CSV" ]]; then
    echo "ERROR: TRACE_CSV is required for MODE=$MODE" >&2
    echo "Example: TRACE_CSV=output/.../*_hcrl_zk_schedule_trace.csv MODE=prepare-inputs bash scripts/run_zk_vos_batch_scalability.sh" >&2
    exit 1
  fi
fi

if [[ "$MODE" == "verify" || "$MODE" == "full" ]]; then
  if [[ -z "$VKEY" ]]; then
    echo "ERROR: VKEY=path/to/verification_key.json is required for MODE=$MODE" >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [[ -n "$TRACE_CSV" ]]; then EXTRA_ARGS+=(--trace-csv "$TRACE_CSV"); fi
if [[ -n "$VKEY" ]]; then EXTRA_ARGS+=(--vkey "$VKEY"); fi
if [[ -n "$WASM" ]]; then EXTRA_ARGS+=(--wasm "$WASM"); fi
if [[ -n "$ZKEY" ]]; then EXTRA_ARGS+=(--zkey "$ZKEY"); fi

python tools/zk_vos_batch_scalability.py \
  --mode "$MODE" \
  --sizes "$SIZES" \
  --proof-dir "$PROOF_DIR" \
  --public-dir "$PUBLIC_DIR" \
  --out-dir "$OUT_DIR" \
  --repeat-trace-rows \
  --repeat-fixtures \
  "${EXTRA_ARGS[@]}"
