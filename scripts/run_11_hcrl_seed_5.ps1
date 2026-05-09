# 5-seed HCRL-Oracle formal experiment
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
$repo = Get-Location
Set-Location -Path "TCO-DRL_with baseline"
$group = "hcrl_seed_" + (Get-Date -Format "yy_M_d_HH_mm")
$seeds = @(6,7,8,9,10)
foreach ($s in $seeds) {
  Write-Host "================ Running HCRL seed $s ================"
  python main.py `
    --Scenario rl_harder `
    --Use_RA_DDQN `
    --Use_PB_SafeDQN `
    --Use_COBRA `
    --Use_HCRL `
    --Oracles_Per_Type 10 `
    --Epoch 30 `
    --Request_Num 6000 `
    --Reward_Scale 2.0 `
    --Reward_Clip 2.0 `
    --Dqn_lr 0.0015 `
    --RA_lr 0.0012 `
    --COBRA_lr 0.0014 `
    --HCRL_lr 0.0014 `
    --HCRL_Mode_lr 0.0012 `
    --Dqn_batch_size 128 `
    --Dqn_memory_size 10000 `
    --Dqn_epsilon_increment 0.0008 `
    --Seed $s `
    --Run_Tag ($group + "_seed" + $s)
}
Set-Location -Path $repo
python .\scripts\aggregate_seed_results.py --output_dir ".\TCO-DRL_with baseline\output"
