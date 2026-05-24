param(
  [string]$Trace = ".\experiments_real_trace\data\real_oracle_trace.csv",
  [string]$Out = ".\collusion_gnn_experiments\output",
  [string]$Seeds = "3,4,5,6,7",
  [int]$Requests = 12000,
  [int]$Oracles = 120,
  [double]$MaliciousRatio = 0.30,
  [int]$WindowSize = 120,
  [int]$Epochs = 80
)

Write-Host "===== HCRL collusion-aware GNN experiment ====="
Write-Host "Trace: $Trace"
Write-Host "Output: $Out"

python .\collusion_gnn_experiments\run_collusion_gnn_experiments.py `
  --trace $Trace `
  --out $Out `
  --seeds $Seeds `
  --requests $Requests `
  --oracles $Oracles `
  --malicious-ratio $MaliciousRatio `
  --window-size $WindowSize `
  --epochs $Epochs

if ($LASTEXITCODE -ne 0) { throw "Collusion GNN experiment failed with exit code $LASTEXITCODE" }

python .\collusion_gnn_experiments\plot_collusion_gnn_results.py `
  --input $Out `
  --out "$Out\figures"

if ($LASTEXITCODE -ne 0) { throw "Plotting failed with exit code $LASTEXITCODE" }

Write-Host "Done. Results are under: $Out"
