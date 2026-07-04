param(
    [string]$TracePath = ".\experiments_real_trace\data\real_oracle_trace.csv",
    [string]$OutDir = ".\audit_reputation_recovery_experiments\output",
    [string]$Seeds = "3,4,5,6,7",
    [int]$Requests = 12000,
    [int]$Oracles = 120,
    [double]$MaliciousRatio = 0.30,
    [double]$AttackOnsetRatio = 0.25,
    [double]$AttackEndRatio = 0.65,
    [string]$Scenarios = "reputation_poisoning,sleeper_attack,collusion_shift,burst_attack,intermittent_evasion,gradual_drift,mev_based_manipulation,strategic_economic_collusion,cross_chain_latency_attack,real_world_oracle_cartel,bribery_staking_game,liquidation_front_running"
)

Write-Host "[1/3] Running audit reputation recovery experiment..." -ForegroundColor Cyan
python .\audit_reputation_recovery_experiments\run_audit_reputation_recovery.py `
  --trace $TracePath `
  --out $OutDir `
  --seeds $Seeds `
  --requests $Requests `
  --oracles $Oracles `
  --malicious-ratio $MaliciousRatio `
  --attack-onset-ratio $AttackOnsetRatio `
  --attack-end-ratio $AttackEndRatio `
  --scenarios $Scenarios

Write-Host "[2/3] Plotting representative paper figure..." -ForegroundColor Cyan
python .\audit_reputation_recovery_experiments\plot_reputation_recovery_curves.py `
  --curve-csv "$OutDir\audit_reputation_recovery_curve.csv" `
  --summary-csv "$OutDir\audit_reputation_recovery_summary_mean_std.csv" `
  --out "$OutDir\audit_reputation_recovery_figure.png"

Write-Host "[3/3] Plotting all-attack curve figure..." -ForegroundColor Cyan
python .\audit_reputation_recovery_experiments\plot_all_attack_recovery_curves.py `
  --curve-csv "$OutDir\audit_reputation_recovery_curve.csv" `
  --summary-csv "$OutDir\audit_reputation_recovery_summary_mean_std.csv" `
  --out "$OutDir\audit_reputation_recovery_all_attacks_figure.png"

Write-Host "[Done] outputs written to $OutDir" -ForegroundColor Green
