# HCRL-Oracle ablation study
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
Set-Location -Path "TCO-DRL_with baseline"
$common = @(
  "--Scenario", "rl_harder",
  "--Use_RA_DDQN",
  "--Use_PB_SafeDQN",
  "--Use_COBRA",
  "--Use_HCRL",
  "--Oracles_Per_Type", "10",
  "--Epoch", "30",
  "--Request_Num", "6000",
  "--Reward_Scale", "2.0",
  "--Reward_Clip", "2.0",
  "--Dqn_lr", "0.0015",
  "--RA_lr", "0.0012",
  "--COBRA_lr", "0.0014",
  "--HCRL_lr", "0.0014",
  "--HCRL_Mode_lr", "0.0012",
  "--Dqn_batch_size", "128",
  "--Dqn_memory_size", "10000",
  "--Dqn_epsilon_increment", "0.0008"
)
$ablations = @(
  @{tag="hcrl_full"; extra=@()},
  @{tag="hcrl_no_teacher"; extra=@("--HCRL_No_Teacher")},
  @{tag="hcrl_no_decoupled"; extra=@("--HCRL_No_Decoupled_Reward")},
  @{tag="hcrl_no_constrained"; extra=@("--HCRL_No_Constrained")},
  @{tag="hcrl_random_backup"; extra=@("--HCRL_Random_Backup")},
  @{tag="hcrl_fixed_single"; extra=@("--HCRL_Fixed_Single_Mode")},
  @{tag="hcrl_fixed_parallel"; extra=@("--HCRL_Fixed_Parallel_Mode")}
)
foreach ($a in $ablations) {
  Write-Host "================ Running $($a.tag) ================"
  python main.py @common @($a.extra) --Run_Tag $a.tag
}
