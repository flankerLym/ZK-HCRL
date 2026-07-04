param(
  [string]$PowersOfTau = "",
  [string]$Circom = "circom",
  [string]$Snarkjs = "snarkjs"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Circuits = Join-Path $Root "circuits"
$Build = Join-Path $Root "build"
New-Item -ItemType Directory -Force -Path $Build | Out-Null

if (-not $PowersOfTau) {
  throw "Please pass -PowersOfTau path/to/potXX_final.ptau. Example: .\compile_zk_vos_ablation.ps1 -PowersOfTau ..\pot12_final.ptau"
}
if (-not (Test-Path $PowersOfTau)) {
  throw "Powers of Tau file not found: $PowersOfTau"
}

$variants = @(
  @{Name="membership_only"; File="membership_only.circom"},
  @{Name="cost_latency"; File="cost_latency.circom"},
  @{Name="risk"; File="risk.circom"},
  @{Name="audit_update"; File="audit_update.circom"},
  @{Name="full_zk_vos"; File="full_zk_vos.circom"}
)

foreach ($v in $variants) {
  $name = $v.Name
  $file = Join-Path $Circuits $v.File
  $out = Join-Path $Build $name
  New-Item -ItemType Directory -Force -Path $out | Out-Null

  Write-Host "[compile] $name"
  & $Circom $file --r1cs --wasm --sym -o $out

  $r1cs = Join-Path $out "$name.r1cs"
  $zkey0 = Join-Path $out "${name}_0000.zkey"
  $zkeyFinal = Join-Path $out "${name}_final.zkey"
  $vkey = Join-Path $out "verification_key.json"

  & $Snarkjs groth16 setup $r1cs $PowersOfTau $zkey0
  & $Snarkjs zkey contribute $zkey0 $zkeyFinal --name="zk-vos-ablation-$name" -v -e="zk-vos-ablation-fixed-entropy-$name"
  & $Snarkjs zkey export verificationkey $zkeyFinal $vkey
  & $Snarkjs r1cs info $r1cs
}

Write-Host "Ablation circuits compiled under: $Build"
