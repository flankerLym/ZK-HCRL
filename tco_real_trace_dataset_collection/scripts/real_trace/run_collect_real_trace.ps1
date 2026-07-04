param(
    [string]$StartDate = "2024-01-01",
    [string]$EndDate = "2024-01-31",
    [string]$RpcUrl = $env:ETH_RPC_URL,
    [string[]]$BinanceSymbols = @("ETHUSDT", "BTCUSDT", "LINKUSDT", "SOLUSDT"),
    [string[]]$PythSymbols = @("Crypto.ETH/USD", "Crypto.BTC/USD", "Crypto.LINK/USD", "Crypto.SOL/USD"),
    [switch]$SkipPyth,
    [switch]$NoReceipts,
    [switch]$TrustEnv,
    [ValidateSet("auto", "events", "calls")]
    [string]$ChainlinkMode = "auto",
    [int]$HistoricalSampleSec = 60,
    [int]$MaxRetries = 5,
    [double]$RetryBackoffSec = 2.0,
    [double]$RetryMaxBackoffSec = 60.0,
    [int]$SaveEvery = 25,
    [switch]$NoResume,
    [switch]$StrictFailures
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RpcUrl)) {
    Write-Error "RpcUrl is empty. Set `$env:ETH_RPC_URL or pass -RpcUrl."
    exit 1
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$ArgsList
    )
    Write-Host "`n===== $Name ====="
    & python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name, exit code=$LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Force -Path "data/raw/binance" | Out-Null
New-Item -ItemType Directory -Force -Path "data/raw/chainlink" | Out-Null
New-Item -ItemType Directory -Force -Path "data/raw/pyth" | Out-Null
New-Item -ItemType Directory -Force -Path "data/processed" | Out-Null

$networkFlag = @()
if ($TrustEnv) {
    $networkFlag = @("--trust-env")
    Write-Host "[network] TrustEnv enabled: Python may use Windows/system/env proxy settings."
} else {
    Write-Host "[network] TrustEnv disabled: Python ignores Windows/system/env proxy settings. This is recommended for TUN/global VPN mode."
}

$binanceArgs = @(
  "scripts/real_trace/collect_binance_klines.py",
  "--symbols"
) + $BinanceSymbols + @(
  "--interval", "1m",
  "--start", $StartDate,
  "--end", $EndDate,
  "--out", "data/raw/binance"
) + $networkFlag
Invoke-PythonStep -Name "Collect Binance klines" -ArgsList $binanceArgs

$receiptFlag = @()
if (-not $NoReceipts) { $receiptFlag = @("--fetch-receipts") }

$resumeFlag = @()
if ($NoResume) { $resumeFlag = @("--no-resume") }

$failureFlag = @()
if ($StrictFailures) { $failureFlag = @("--strict-failures") }

$chainlinkArgs = @(
  "scripts/real_trace/collect_chainlink_rounds.py",
  "--rpc-url", $RpcUrl,
  "--feeds", "config/chainlink_feeds.yaml",
  "--start", $StartDate,
  "--end", $EndDate,
  "--out", "data/raw/chainlink",
  "--block-chunk", "3000",
  "--mode", $ChainlinkMode,
  "--historical-sample-sec", "$HistoricalSampleSec",
  "--max-retries", "$MaxRetries",
  "--retry-backoff-sec", "$RetryBackoffSec",
  "--retry-max-backoff-sec", "$RetryMaxBackoffSec",
  "--save-every", "$SaveEvery"
) + $receiptFlag + $resumeFlag + $failureFlag + $networkFlag
Invoke-PythonStep -Name "Collect Chainlink rounds" -ArgsList $chainlinkArgs

if (-not $SkipPyth) {
  $pythArgs = @(
    "scripts/real_trace/collect_pyth_benchmarks.py",
    "--symbols"
  ) + $PythSymbols + @(
    "--resolution", "1",
    "--start", $StartDate,
    "--end", $EndDate,
    "--out", "data/raw/pyth"
  ) + $networkFlag
  Invoke-PythonStep -Name "Collect Pyth benchmarks" -ArgsList $pythArgs

  $buildArgs = @(
    "scripts/real_trace/build_real_oracle_trace.py",
    "--chainlink-dir", "data/raw/chainlink",
    "--binance-dir", "data/raw/binance",
    "--pyth-dir", "data/raw/pyth",
    "--out", "data/processed/real_oracle_trace.csv",
    "--deviation-threshold", "0.01",
    "--staleness-threshold-sec", "600",
    "--include-pyth"
  )
  Invoke-PythonStep -Name "Build real oracle trace" -ArgsList $buildArgs
} else {
  $buildArgs = @(
    "scripts/real_trace/build_real_oracle_trace.py",
    "--chainlink-dir", "data/raw/chainlink",
    "--binance-dir", "data/raw/binance",
    "--out", "data/processed/real_oracle_trace.csv",
    "--deviation-threshold", "0.01",
    "--staleness-threshold-sec", "600"
  )
  Invoke-PythonStep -Name "Build real oracle trace" -ArgsList $buildArgs
}

$summaryArgs = @(
  "scripts/real_trace/summarize_real_trace.py",
  "--input", "data/processed/real_oracle_trace.csv",
  "--out", "data/processed/real_oracle_trace_summary.csv"
)
Invoke-PythonStep -Name "Summarize real oracle trace" -ArgsList $summaryArgs
