#!/usr/bin/env bash
set -euo pipefail

START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-01-31}"
RPC_URL="${ETH_RPC_URL:-}"
SKIP_PYTH="${SKIP_PYTH:-0}"
NO_RECEIPTS="${NO_RECEIPTS:-0}"
TRUST_ENV="${TRUST_ENV:-0}"
CHAINLINK_MODE="${CHAINLINK_MODE:-auto}"
HISTORICAL_SAMPLE_SEC="${HISTORICAL_SAMPLE_SEC:-60}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-2.0}"
RETRY_MAX_BACKOFF_SEC="${RETRY_MAX_BACKOFF_SEC:-60.0}"
SAVE_EVERY="${SAVE_EVERY:-25}"
NO_RESUME="${NO_RESUME:-0}"
STRICT_FAILURES="${STRICT_FAILURES:-0}"

if [[ -z "${RPC_URL}" ]]; then
  echo "ETH_RPC_URL is empty. Export ETH_RPC_URL first." >&2
  exit 1
fi

NETWORK_FLAG=()
if [[ "${TRUST_ENV}" == "1" ]]; then
  NETWORK_FLAG=(--trust-env)
  echo "[network] TRUST_ENV=1: Python may use system/env proxies."
else
  echo "[network] TRUST_ENV=0: Python ignores system/env proxies. Recommended for TUN/global VPN mode."
fi

mkdir -p data/raw/binance data/raw/chainlink data/raw/pyth data/processed

python scripts/real_trace/collect_binance_klines.py   --symbols ETHUSDT BTCUSDT LINKUSDT SOLUSDT   --interval 1m   --start "${START_DATE}"   --end "${END_DATE}"   --out data/raw/binance   "${NETWORK_FLAG[@]}"

RECEIPT_FLAG=()
if [[ "${NO_RECEIPTS}" != "1" ]]; then
  RECEIPT_FLAG=(--fetch-receipts)
fi
RESUME_FLAG=()
if [[ "${NO_RESUME}" == "1" ]]; then
  RESUME_FLAG=(--no-resume)
fi
FAILURE_FLAG=()
if [[ "${STRICT_FAILURES}" == "1" ]]; then
  FAILURE_FLAG=(--strict-failures)
fi

python scripts/real_trace/collect_chainlink_rounds.py   --rpc-url "${RPC_URL}"   --feeds config/chainlink_feeds.yaml   --start "${START_DATE}"   --end "${END_DATE}"   --out data/raw/chainlink   --block-chunk 3000   --mode "${CHAINLINK_MODE}"   --historical-sample-sec "${HISTORICAL_SAMPLE_SEC}"   --max-retries "${MAX_RETRIES}"   --retry-backoff-sec "${RETRY_BACKOFF_SEC}"   --retry-max-backoff-sec "${RETRY_MAX_BACKOFF_SEC}"   --save-every "${SAVE_EVERY}"   "${RECEIPT_FLAG[@]}"   "${RESUME_FLAG[@]}"   "${FAILURE_FLAG[@]}"   "${NETWORK_FLAG[@]}"

if [[ "${SKIP_PYTH}" != "1" ]]; then
  python scripts/real_trace/collect_pyth_benchmarks.py     --symbols Crypto.ETH/USD Crypto.BTC/USD Crypto.LINK/USD Crypto.SOL/USD     --resolution 1     --start "${START_DATE}"     --end "${END_DATE}"     --out data/raw/pyth     "${NETWORK_FLAG[@]}"

  python scripts/real_trace/build_real_oracle_trace.py     --chainlink-dir data/raw/chainlink     --binance-dir data/raw/binance     --pyth-dir data/raw/pyth     --out data/processed/real_oracle_trace.csv     --deviation-threshold 0.01     --staleness-threshold-sec 600     --include-pyth
else
  python scripts/real_trace/build_real_oracle_trace.py     --chainlink-dir data/raw/chainlink     --binance-dir data/raw/binance     --out data/processed/real_oracle_trace.csv     --deviation-threshold 0.01     --staleness-threshold-sec 600
fi

python scripts/real_trace/summarize_real_trace.py   --input data/processed/real_oracle_trace.csv   --out data/processed/real_oracle_trace_summary.csv
