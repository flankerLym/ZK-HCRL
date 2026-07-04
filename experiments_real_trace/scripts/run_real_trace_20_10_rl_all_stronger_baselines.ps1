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
    [int]$BaselineEnhanceTopK = 10,
    [double]$BaselineEnhanceRiskWeight = 0.55,
    [double]$BaselineEnhanceRepWeight = 0.25,
    [double]$BaselineEnhanceOntimeWeight = 0.20,
    [double]$BaselineEnhanceCostWeight = 0.08,
    [double]$BaselineEnhanceMaxQDrop = 0.35,
    [string]$BaselineEnhanceTargets = "DQN,RA-DDQN,PB-SafeDQN,COBRA-Oracle",
    [string]$OutputRoot = "experiments_real_trace\output"
)

$ErrorActionPreference = "Stop"
$runner = "experiments_real_trace\real_trace_main.py"
$trainOut = Join-Path $OutputRoot "train_rl_all_stronger_baselines"
$testOut  = Join-Path $OutputRoot "test_rl_all_stronger_baselines"
if ($WorkloadMode -eq "full_trace") {
    $trainOut = Join-Path $OutputRoot "train_rl_all_stronger_baselines_fulltrace"
    $testOut  = Join-Path $OutputRoot "test_rl_all_stronger_baselines_fulltrace"
}
New-Item -ItemType Directory -Force -Path $trainOut | Out-Null
New-Item -ItemType Directory -Force -Path $testOut | Out-Null

$methods = @("DQN", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle")

Write-Host "===== Real-trace matched RL: stronger-but-fair baseline training ====="
Write-Host "Mode=$WorkloadMode, RequestNum=$RequestNum, BaselineEnhanceTopK=$BaselineEnhanceTopK, RiskWeight=$BaselineEnhanceRiskWeight"
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
  --Enhance_Baseline_Safety `
  --Baseline_Enhance_Targets $BaselineEnhanceTargets `
  --Baseline_Enhance_TopK $BaselineEnhanceTopK `
  --Baseline_Enhance_Risk_Weight $BaselineEnhanceRiskWeight `
  --Baseline_Enhance_Rep_Weight $BaselineEnhanceRepWeight `
  --Baseline_Enhance_Ontime_Weight $BaselineEnhanceOntimeWeight `
  --Baseline_Enhance_Cost_Weight $BaselineEnhanceCostWeight `
  --Baseline_Enhance_Max_Q_Drop $BaselineEnhanceMaxQDrop `
  --Methods $methods `
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
  --Run_Tag "strongbase_train"

if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
$trainRun = Get-ChildItem $trainOut -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $trainRun) { throw "No train run directory found under $trainOut" }
Write-Host "Train run directory: $($trainRun.FullName)"

Write-Host "===== Real-trace matched RL: stronger-but-fair baseline test ====="
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
  --Enhance_Baseline_Safety `
  --Baseline_Enhance_Targets $BaselineEnhanceTargets `
  --Baseline_Enhance_TopK $BaselineEnhanceTopK `
  --Baseline_Enhance_Risk_Weight $BaselineEnhanceRiskWeight `
  --Baseline_Enhance_Rep_Weight $BaselineEnhanceRepWeight `
  --Baseline_Enhance_Ontime_Weight $BaselineEnhanceOntimeWeight `
  --Baseline_Enhance_Cost_Weight $BaselineEnhanceCostWeight `
  --Baseline_Enhance_Max_Q_Drop $BaselineEnhanceMaxQDrop `
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
  --Load_Weights "DQN=$($trainRun.FullName)" "RA-DDQN=$($trainRun.FullName)" "PB-SafeDQN=$($trainRun.FullName)" "COBRA-Oracle=$($trainRun.FullName)" "HCRL-Oracle=$($trainRun.FullName)" `
  --Output_Dir $testOut `
  --Run_Tag "strongbase_test"

if ($LASTEXITCODE -ne 0) { throw "Test failed with exit code $LASTEXITCODE" }
Write-Host "Done. Stronger-baseline outputs are under: $trainOut and $testOut"
