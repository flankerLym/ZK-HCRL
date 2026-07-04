#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cd experiments/zk_vos
npm install
bash scripts/run_full_pipeline.sh
