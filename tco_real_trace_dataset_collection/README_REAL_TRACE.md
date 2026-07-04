# Real-Trace Dataset Collection for HCRL-Oracle

This package adds a standalone real-data collection pipeline for the TCO-DRL/HCRL project.

It builds `data/processed/real_oracle_trace.csv` from:

1. Chainlink price-feed update events on Ethereum
2. Binance one-minute kline reference prices
3. Optional Pyth benchmark prices
4. Optional Ethereum gas/receipt data from your RPC node

The goal is not to claim that public data provides ground-truth malicious oracle labels. Instead, this pipeline creates a **real-trace-driven oracle scheduling benchmark**:

- real price-feed update times become workload traces;
- real market prices become validation references;
- price deviation, staleness, and confidence become risk features;
- anomaly labels are threshold-based and reproducible.

## Install

```bash
pip install -r requirements_real_trace.txt
```

For Chainlink collection, set an Ethereum RPC URL:

```powershell
$env:ETH_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
```

or Linux/macOS:

```bash
export ETH_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
```

## Quick start: 30-day ETH/BTC/LINK trace

PowerShell:

```powershell
.\scripts\real_trace\run_collect_real_trace.ps1 `
  -StartDate "2024-01-01" `
  -EndDate "2024-01-31" `
  -RpcUrl $env:ETH_RPC_URL
```

Linux/macOS:

```bash
bash scripts/real_trace/run_collect_real_trace.sh 2024-01-01 2024-01-31 "$ETH_RPC_URL"
```

The final output is:

```text
data/processed/real_oracle_trace.csv
data/processed/real_oracle_trace_summary.csv
```

## Manual commands

### 1. Binance reference prices

```bash
python scripts/real_trace/collect_binance_klines.py   --symbols ETHUSDT BTCUSDT LINKUSDT SOLUSDT   --interval 1m   --start 2024-01-01   --end 2024-01-31   --out data/raw/binance
```

### 2. Chainlink feed updates

```bash
python scripts/real_trace/collect_chainlink_rounds.py   --rpc-url "$ETH_RPC_URL"   --feeds config/chainlink_feeds.yaml   --start 2024-01-01   --end 2024-01-31   --out data/raw/chainlink   --fetch-receipts
```

Notes:

- Feed addresses in `config/chainlink_feeds.yaml` are examples for Ethereum mainnet.
- Always verify addresses against the current Chainlink docs before paper submission.
- Some RPC providers restrict historical log range size. Use `--block-chunk 2000` or smaller if needed.

### 3. Optional Pyth benchmark prices

```bash
python scripts/real_trace/collect_pyth_benchmarks.py   --symbols Crypto.ETH/USD Crypto.BTC/USD Crypto.LINK/USD Crypto.SOL/USD   --resolution 1   --start 2024-01-01   --end 2024-01-31   --out data/raw/pyth
```

### 4. Build HCRL-compatible trace

```bash
python scripts/real_trace/build_real_oracle_trace.py   --chainlink-dir data/raw/chainlink   --binance-dir data/raw/binance   --pyth-dir data/raw/pyth   --out data/processed/real_oracle_trace.csv   --deviation-threshold 0.01   --staleness-threshold-sec 600   --include-pyth
```

### 5. Summarize

```bash
python scripts/real_trace/summarize_real_trace.py   --input data/processed/real_oracle_trace.csv   --out data/processed/real_oracle_trace_summary.csv
```

## Output schema

`real_oracle_trace.csv` contains:

| Column | Meaning |
|---|---|
| timestamp | UTC timestamp of the oracle update / request |
| asset | Asset pair such as ETH/USD |
| source | chainlink or pyth |
| oracle_id | Source-asset identifier |
| oracle_price | Oracle price |
| reference_price | Binance reference price aligned by timestamp |
| deviation | Absolute relative price deviation |
| staleness | Seconds since the previous update of the same source-asset |
| confidence_ratio | Pyth confidence / price when available, otherwise NaN |
| gas_cost | Estimated Ethereum gas cost in ETH when available |
| latency | Proxy latency in seconds, currently update interval / staleness |
| validation_success | 1 if deviation and staleness pass thresholds |
| anomaly_label | normal / suspicious / anomalous |
| service_type | Numeric service type id derived from asset |

## Recommended paper wording

> We construct a real-trace-driven oracle scheduling benchmark from public Chainlink/Pyth price-feed updates and Binance reference prices. Because public oracle traces do not provide ground-truth malicious node labels, we define risky oracle states using reproducible thresholds on cross-market price deviation, staleness, and confidence intervals.

## Important limitations

- This pipeline does not identify real malicious Chainlink/Pyth nodes.
- Chainlink public feeds often expose aggregated on-chain updates rather than raw off-chain node responses.
- The resulting labels are anomaly/risk labels, not legally or cryptographically verified malicious-node labels.
- Historical Pyth confidence values may depend on the endpoint you use. The benchmark collector uses Pyth benchmark prices; confidence can be added from Hermes snapshots or archival services if available.

## Network mode: TUN/global VPN vs explicit proxy

The collectors now default to `trust_env=False`, so Python `requests` and Web3 RPC calls ignore Windows system proxy settings and environment proxies. This is recommended when your VPN runs in TUN/global mode.

PowerShell default for TUN/global VPN:

```powershell
.\scripts\real_trace\run_collect_real_trace.ps1 `
  -StartDate "2024-01-01" `
  -EndDate "2024-01-02" `
  -RpcUrl "https://eth-mainnet.g.alchemy.com/v2/YOUR_NEW_KEY" `
  -NoReceipts
```

If you intentionally want Python to use Windows/system/env proxy settings, add `-TrustEnv`:

```powershell
.\scripts\real_trace\run_collect_real_trace.ps1 `
  -StartDate "2024-01-01" `
  -EndDate "2024-01-02" `
  -RpcUrl "https://eth-mainnet.g.alchemy.com/v2/YOUR_NEW_KEY" `
  -TrustEnv
```

The PowerShell wrapper now stops immediately if Binance, Chainlink, or Pyth collection fails. This prevents producing a misleading tiny `real_oracle_trace.csv` after partial network failures.

## v3 Chainlink collection fallback

If Alchemy returns `400 Client Error: Bad Request` during `AnswerUpdated.get_logs`, the collector now automatically falls back to historical `latestRoundData` calls. This produces one sampled Chainlink state per feed per minute by default. The resulting CSV is still valid for real-trace-driven workload/validation experiments because it records the sampled request time, the latest on-chain oracle price at that historical block, and the true feed update time used to compute staleness.

PowerShell options:

```powershell
# Default: try events, then fallback to historical calls
.\scripts\real_trace\run_collect_real_trace.ps1 `
  -StartDate "2024-01-01" `
  -EndDate "2024-01-02" `
  -RpcUrl "YOUR_ALCHEMY_RPC" `
  -NoReceipts

# Force historical-call mode if eth_getLogs keeps failing
.\scripts\real_trace\run_collect_real_trace.ps1 `
  -StartDate "2024-01-01" `
  -EndDate "2024-01-02" `
  -RpcUrl "YOUR_ALCHEMY_RPC" `
  -NoReceipts `
  -ChainlinkMode calls `
  -HistoricalSampleSec 60
```

Notes:
- `-NoReceipts` is recommended for the first run because historical sampling does not need transaction receipts.
- If historical calls fail, your RPC provider may not support archive/historical state for the selected date.
- Binance files downloaded successfully can be kept; delete only `data/raw/chainlink` and `data/processed` before rerunning the Chainlink/build steps.

### Binance timestamp unit compatibility

Some newer Binance Data Vision kline archives may store `open_time`/`close_time` in microseconds instead of milliseconds. The collector now detects timestamp units automatically and avoids pandas `OutOfBoundsDatetime` errors such as year 58217.

## v5 robustness update: retry / skip / resume

For long Chainlink historical-call collection, the collector now supports:

- `--max-retries`: retry each historical RPC call before considering it failed.
- `--retry-backoff-sec` and `--retry-max-backoff-sec`: exponential backoff controls.
- `--save-every`: periodically flush partial Chainlink CSVs to disk.
- resume by default: if `data/raw/chainlink/<ASSET>.csv` already exists, the collector skips sampled blocks already present in that file.
- skip failed sample points by default: if a block still fails after all retries, it is logged under `data/raw/chainlink/failed_points/` and the run continues.
- `--strict-failures`: stop on a failed sample point instead of skipping.
- `--no-resume`: ignore existing partial files and recollect from scratch.

PowerShell example for a more robust 30-day, 10-minute, 3-asset collection:

```powershell
.\scripts\real_trace\run_collect_real_trace.ps1 `
  -StartDate "2026-04-01" `
  -EndDate "2026-05-01" `
  -RpcUrl "<YOUR_ALCHEMY_RPC_URL>" `
  -NoReceipts `
  -SkipPyth `
  -ChainlinkMode calls `
  -HistoricalSampleSec 600 `
  -BinanceSymbols @("ETHUSDT", "BTCUSDT", "LINKUSDT") `
  -MaxRetries 8 `
  -RetryBackoffSec 2 `
  -RetryMaxBackoffSec 60 `
  -SaveEvery 20
```

If the run is interrupted, run the same command again. It will resume from the existing partial Chainlink CSV files.
