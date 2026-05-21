param(
    [string]$TraceCsv = "experiments_real_trace\data\real_oracle_trace.csv"
)
$ErrorActionPreference = "Stop"
python experiments_real_trace\real_trace_main.py `
  --Real_Trace_Path $TraceCsv `
  --Real_Trace_Split train `
  --Real_Trace_Train_Days 20 `
  --No_Real_Trace_Auto_Request_Num `
  --Real_Trace_Workload_Mode matched `
  --Real_Trace_Risk_Strength 0.20 `
  --Real_Trace_Feature_Blend 0.15 `
  --Real_Trace_Metric_Blend 0.20 `
  --Real_Trace_Latency_Max_Penalty 0.25 `
  --Methods HCRL-Oracle `
  --Epoch 1 `
  --Seed 3 `
  --Request_Num 60 `
  --Scenario rl_harder `
  --Oracles_Per_Type 5 `
  --Malicious_Ratio 0.30 `
  --State_Mode enhanced `
  --Reward_Mode risk_aware `
  --Success_Mode validation_aware `
  --Action_Mask_Mode type `
  --Use_GNN_Encoder `
  --Output_Dir experiments_real_trace\output\smoke_matched `
  --Run_Tag "smoke_realtrace_matched"
