param(
    [string]$TracePath = ".\experiments_real_trace\data\real_oracle_trace.csv",
    [string]$OutDir = ".\audit_reputation_recovery_experiments\output",
    [string]$Seeds = "3,4,5,6,7",
    [int]$Requests = 12000,
    [int]$Oracles = 120,
    [double]$MaliciousRatio = 0.30,
    [double]$AttackOnsetRatio = 0.25,
    [double]$AttackEndRatio = 0.65
)

Write-Host "[1/2] Running audit reputation recovery experiment..." -ForegroundColor Cyan
python .\audit_reputation_recovery_experiments\run_audit_reputation_recovery.py `
  --trace $TracePath `
  --out $OutDir `
  --seeds $Seeds `
  --requests $Requests `
  --oracles $Oracles `
  --malicious-ratio $MaliciousRatio `
  --attack-onset-ratio $AttackOnsetRatio `
  --attack-end-ratio $AttackEndRatio

Write-Host "[2/3] Plotting representative paper figure..." -ForegroundColor Cyan
python .\audit_reputation_recovery_experiments\plot_reputation_recovery_curves.py `
  --curve-csv "$OutDir\audit_reputation_recovery_curve.csv" `
  --summary-csv "$OutDir\audit_reputation_recovery_summary_mean_std.csv" `
  --out "$OutDir\audit_reputation_recovery_figure.png"

Write-Host "[3/3] Plotting all-six-attack curve figure..." -ForegroundColor Cyan
python .\audit_reputation_recovery_experiments\plot_all_attack_recovery_curves.py `
  --curve-csv "$OutDir\audit_reputation_recovery_curve.csv" `
  --summary-csv "$OutDir\audit_reputation_recovery_summary_mean_std.csv" `
  --out "$OutDir\audit_reputation_recovery_all_attacks_figure.png"

Write-Host "[Done] outputs written to $OutDir" -ForegroundColor Green
