param(
    [string]$TraceCsv = "experiments_real_trace\data\real_oracle_trace.csv",
    [string]$TrainRunDir = "",
    [int]$Seed = 3,
    [int]$OraclesPerType = 40,
    [double]$MaliciousRatio = 0.30,
    [int]$TrainDays = 20,
    [int]$RequestNum = 6000,
    [ValidateSet("matched", "full_trace")]
    [string]$WorkloadMode = "matched",
    [double]$TraceRiskStrength = 0.20,
    [double]$TraceFeatureBlend = 0.15,
    [double]$TraceMetricBlend = 0.20,
    [double]$TraceLatencyMaxPenalty = 0.25,
    [ValidateSet("guarded", "greedy", "softmax")]
    [string]$HCRLEvalModePolicy = "guarded",
    [double]$HCRLEvalModeTemperature = 1.25,
    [double]$HCRLEvalParallelMaxRate = 0.75,
    [double]$HCRLEvalQMargin = 0.05,
    [string]$OutputRoot = "experiments_real_trace\output"
)

$ErrorActionPreference = "Stop"
$runner = "experiments_real_trace\real_trace_main.py"
$testOut = Join-Path $OutputRoot "test_guarded"
if ($WorkloadMode -eq "full_trace") {
    $testOut = Join-Path $OutputRoot "test_full_guarded"
}
New-Item -ItemType Directory -Force -Path $testOut | Out-Null

if ([string]::IsNullOrWhiteSpace($TrainRunDir)) {
    $candidateRoots = @(
        (Join-Path $OutputRoot "train_rl_all_matched"),
        (Join-Path $OutputRoot "train_rl_all"),
        (Join-Path $OutputRoot "train_hcrl_matched"),
        (Join-Path $OutputRoot "train_hcrl")
    ) | Where-Object { Test-Path $_ }
    $latest = $candidateRoots | ForEach-Object { Get-ChildItem $_ -Directory -ErrorAction SilentlyContinue } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw "No training run found. Pass -TrainRunDir explicitly." }
    $TrainRunDir = $latest.FullName
}
if (!(Test-Path $TrainRunDir)) { throw "TrainRunDir not found: $TrainRunDir" }
Write-Host "Using train run: $TrainRunDir"

$methods = @("DQN", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle")
Write-Host "===== Real-trace test from existing train: guarded HCRL mode eval ====="
Write-Host "Mode=$WorkloadMode, RequestNum=$RequestNum, HCRLEvalModePolicy=$HCRLEvalModePolicy, ParallelMax=$HCRLEvalParallelMaxRate"

python $runner `
  --Real_Trace_Path $TraceCsv `
  --Real_Trace_Split test `
  --Real_Trace_Train_Days $TrainDays `
  --No_Real_Trace_Auto_Request_Num `
  --Real_Trace_Workload_Mode $WorkloadMode `
  --Real_Trace_Risk_Strength $TraceRiskStrength `
  --Real_Trace_Feature_Blend $TraceFeatureBlend `
  --Real_Trace_Metric_Blend $TraceMetricBlend `
  --Real_Trace_Latency_Max_Penalty $TraceLatencyMaxPenalty `
  --HCRL_Eval_Mode_Policy $HCRLEvalModePolicy `
  --HCRL_Eval_Mode_Temperature $HCRLEvalModeTemperature `
  --HCRL_Eval_Parallel_Max_Rate $HCRLEvalParallelMaxRate `
  --HCRL_Eval_Q_Margin $HCRLEvalQMargin `
  --Methods $methods `
  --Epoch 1 `
  --Seed $Seed `
  --Request_Num $RequestNum `
  --Scenario rl_harder `
  --Oracles_Per_Type $OraclesPerType `
  --Malicious_Ratio $MaliciousRatio `
  --Malicious_Placement balanced `
  --State_Mode enhanced `
  --Reward_Mode risk_aware `
  --Success_Mode validation_aware `
  --Action_Mask_Mode type `
  --Use_GNN_Encoder `
  --Eval_Only `
  --Greedy_Eval `
  --Load_Weights "DQN=$TrainRunDir" "RA-DDQN=$TrainRunDir" "PB-SafeDQN=$TrainRunDir" "COBRA-Oracle=$TrainRunDir" "HCRL-Oracle=$TrainRunDir" `
  --Output_Dir $testOut `
  --Run_Tag "test_guarded"

if ($LASTEXITCODE -ne 0) {
    throw "real_trace_main.py failed with exit code $LASTEXITCODE"
}
Write-Host "Done. Test outputs are under: $testOut"
