# Formal 5-seed COBRA-Oracle experiment for paper tables.
# Run from repository root: .\scripts\run_06_seed_cobra_formal_5seeds.ps1
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\TCO-DRL_with baseline"

$RUN_GROUP = "seed_formal_" + (Get-Date -Format "yy_M_d_HH_mm")
$OUTDIR = "output\$RUN_GROUP"
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

$COMMON = @(
    "--Scenario", "rl_harder",
    "--Use_RA_DDQN",
    "--Use_PB_SafeDQN",
    "--Use_COBRA",
    "--Oracles_Per_Type", "10",
    "--Epoch", "30",
    "--Request_Num", "6000",
    "--Reward_Scale", "2.0",
    "--Reward_Clip", "2.0",
    "--Dqn_lr", "0.0015",
    "--RA_lr", "0.0012",
    "--COBRA_lr", "0.0014",
    "--Dqn_batch_size", "128",
    "--Dqn_memory_size", "10000",
    "--Dqn_epsilon_increment", "0.0008",
    "--Output_Dir", $OUTDIR
)

foreach ($seed in 6,7,8,9,10) {
    Write-Host "================ Seed $seed ================"
    python main.py @COMMON --Seed $seed --Run_Tag "formal_seed$seed"
}

Set-Location "$PSScriptRoot\.."
python .\scripts\aggregate_seed_results.py --output_dir ".\TCO-DRL_with baseline\$OUTDIR"
python .\scripts\plot_seed_results_cn.py --summary_csv ".\TCO-DRL_with baseline\$OUTDIR\seed_mean_std.csv" --all_csv ".\TCO-DRL_with baseline\$OUTDIR\seed_all_runs.csv" --output_dir ".\TCO-DRL_with baseline\$OUTDIR\seed_figs"

Write-Host "Formal seed experiment finished. Results folder: TCO-DRL_with baseline\$OUTDIR"
