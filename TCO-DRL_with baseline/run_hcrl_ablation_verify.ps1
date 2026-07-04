# run_hcrl_ablation_verify.ps1
# 放到 TCO-DRL_with baseline 根目录运行
# powershell -ExecutionPolicy Bypass -File ".\run_hcrl_ablation_verify.ps1" -DisablePSReadLine
# 加 -RunStrictCost 可额外跑 strict-cost 对比

param(
  [int]$Seed = 3,
  [int]$Epoch = 30,
  [int]$RequestNum = 6000,
  [int]$OraclesPerType = 10,
  [string]$OutputDir = "E:\tco_ablation_verify_lr0008",
  [switch]$RunStrictCost,
  [switch]$DisablePSReadLine
)

$ErrorActionPreference = "Stop"
if ($DisablePSReadLine) { Remove-Module PSReadLine -ErrorAction SilentlyContinue }

if (-not (Test-Path ".\main.py")) {
  Write-Host "[Error] main.py not found. Run this under TCO-DRL_with baseline."
  exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Base = @(
  "main.py",
  "--Methods", "DQN", "HCRL-Oracle",
  "--Scenario", "rl_harder",
  "--Use_Audit_Reputation",
  "--Epoch", "$Epoch",
  "--Request_Num", "$RequestNum",
  "--Oracles_Per_Type", "$OraclesPerType",
  "--Reward_Scale", "3.0",
  "--Reward_Clip", "3.0",
  "--HCRL_lr", "0.0008",
  "--HCRL_Mode_lr", "0.000616",
  "--Output_Dir", "$OutputDir",
  "--Seed", "$Seed"
)

$Exps = @(
  @{Name="HCRL-Full";      Tag="abl_lr8_full_s$Seed";           Extra=@()},
  @{Name="w/o GNN";        Tag="abl_lr8_no_gnn_s$Seed";         Extra=@("--Disable_GNN_Encoder")},
  @{Name="w/o Audit";      Tag="abl_lr8_no_audit_s$Seed";       Extra=@("--Disable_Audit_Reputation")},
  @{Name="Random Backup";  Tag="abl_lr8_random_backup_s$Seed";  Extra=@("--HCRL_Random_Backup")},
  @{Name="Fixed Single";   Tag="abl_lr8_fixed_single_s$Seed";   Extra=@("--HCRL_Fixed_Single_Mode")},
  @{Name="Fixed Parallel"; Tag="abl_lr8_fixed_parallel_s$Seed"; Extra=@("--HCRL_Fixed_Parallel_Mode")}
)

if ($RunStrictCost) {
  $Exps += @(
    @{Name="HCRL-Full strict-cost";      Tag="abl_lr8_full_strictcost_s$Seed";           Extra=@("--HCRL_Parallel_Cost_Discount","1.0")},
    @{Name="Random Backup strict-cost";  Tag="abl_lr8_random_backup_strictcost_s$Seed";  Extra=@("--HCRL_Random_Backup","--HCRL_Parallel_Cost_Discount","1.0")},
    @{Name="Fixed Single strict-cost";   Tag="abl_lr8_fixed_single_strictcost_s$Seed";   Extra=@("--HCRL_Fixed_Single_Mode","--HCRL_Parallel_Cost_Discount","1.0")},
    @{Name="Fixed Parallel strict-cost"; Tag="abl_lr8_fixed_parallel_strictcost_s$Seed"; Extra=@("--HCRL_Fixed_Parallel_Mode","--HCRL_Parallel_Cost_Discount","1.0")}
  )
}

Write-Host "============================================================"
Write-Host "HCRL Ablation Verification"
Write-Host "OutputDir=$OutputDir Seed=$Seed Epoch=$Epoch"
Write-Host "HCRL_lr=0.0008 HCRL_Mode_lr=0.000616"
Write-Host "StrictCost=$RunStrictCost"
Write-Host "============================================================"

foreach ($e in $Exps) {
  Write-Host ""
  Write-Host "============================================================"
  Write-Host "[Run] $($e.Name)"
  Write-Host "[Tag] $($e.Tag)"
  Write-Host "============================================================"

  $ArgsList = $Base + @("--Run_Tag", $e.Tag) + $e.Extra
  python @ArgsList

  if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] failed: $($e.Name)"
    exit $LASTEXITCODE
  }
}

Write-Host ""
Write-Host "[Done] All experiments finished."

# 打包轻量结果
$ZipName = ".\hcrl_ablation_verify_lr0008_seed${Seed}_epoch${Epoch}_light.zip"
$files = Get-ChildItem $OutputDir -Recurse -File | Where-Object {
  $_.Name -like "*.txt" -or
  $_.Name -like "*_final_results.csv" -or
  $_.Name -like "*_final_results.json"
}
if ($files.Count -gt 0) {
  Compress-Archive -Path $files.FullName -DestinationPath $ZipName -Force
  Write-Host "[Saved] $ZipName"
} else {
  Write-Host "[Warning] No result files found."
}
