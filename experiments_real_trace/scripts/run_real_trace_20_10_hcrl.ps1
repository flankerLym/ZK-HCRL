param(
    [string]$TraceCsv = "experiments_real_trace\data\real_oracle_trace.csv",
    [int]$Epoch = 30,
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
$trainOut = Join-Path $OutputRoot "train_hcrl_matched"
$testOut  = Join-Path $OutputRoot "test_hcrl_matched"
if ($WorkloadMode -eq "full_trace") {
    $trainOut = Join-Path $OutputRoot "train_hcrl_fulltrace"
    $testOut  = Join-Path $OutputRoot "test_hcrl_fulltrace"
}
New-Item -ItemType Directory -Force -Path $trainOut | Out-Null
New-Item -ItemType Directory -Force -Path $testOut | Out-Null

Write-Host "===== Real-trace matched HCRL train: first $TrainDays days ====="
python $runner `
  --Real_Trace_Path $TraceCsv `
  --Real_Trace_Split train `
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
  --Methods HCRL-Oracle `
  --Epoch $Epoch `
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
  --Output_Dir $trainOut `
  --Run_Tag "realtrace_${WorkloadMode}_20d_train_hcrl"

$trainRun = Get-ChildItem $trainOut -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $trainRun) { throw "No train run directory found under $trainOut" }
Write-Host "Train run directory: $($trainRun.FullName)"

Write-Host "===== Real-trace matched HCRL test: last 10 days, greedy primary/backup + guarded mode eval ====="
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
  --Methods HCRL-Oracle `
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
  --Load_Weights "HCRL-Oracle=$($trainRun.FullName)" `
  --Output_Dir $testOut `
  --Run_Tag "realtrace_${WorkloadMode}_10d_test_hcrl"

Write-Host "Done. Test outputs are under: $testOut"
