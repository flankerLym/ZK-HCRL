<#
Run attack-scenario stress tests with pretrained weights.

Usage examples:
  .\scripts\run_attack_scenarios_loaded_weights.ps1 `
    -HCRLWeightDir "output\your_train_run" `
    -DQNWeight "output\your_train_run\DQN.npz"

  .\scripts\run_attack_scenarios_loaded_weights.ps1 `
    -HCRLModeWeight "output\run\HCRL_Mode.npz" `
    -HCRLPrimaryWeight "output\run\HCRL_Primary.npz" `
    -HCRLBackupWeight "output\run\HCRL_Backup.npz"
#>
param(
    [string[]]$Methods = @("DQN", "HCRL-Oracle"),
    [int[]]$Seeds = @(3),
    [string[]]$AttackProfiles = @("mild", "stealth", "mixed", "severe", "burst"),
    [double[]]$AttackIntensities = @(1.0),
    [double[]]$MaliciousRatios = @(0.30, 0.40, 0.50),
    [int]$Epoch = 10,
    [int]$RequestNum = 6000,
    [int]$OraclesPerType = 40,
    [string]$Scenario = "rl_harder",
    [string]$OutputDir = "output",

    # Weight paths. HCRLWeightDir is a run folder containing HCRL_Mode.npz,
    # HCRL_Primary.npz and HCRL_Backup.npz. Individual paths override the dir.
    [string]$DQNWeight = "",
    [string]$PPOWeight = "",
    [string]$RADDQNWeight = "",
    [string]$PBSafeDQNWeight = "",
    [string]$COBRAWeight = "",
    [string]$HCRLWeightDir = "",
    [string]$HCRLModeWeight = "",
    [string]$HCRLPrimaryWeight = "",
    [string]$HCRLBackupWeight = ""
)

function Add-WeightArg {
    param([System.Collections.ArrayList]$List, [string]$Name, [string]$Path)
    if (-not [string]::IsNullOrWhiteSpace($Path)) {
        [void]$List.Add("$Name=$Path")
    }
}

$weightSpecs = New-Object System.Collections.ArrayList
Add-WeightArg $weightSpecs "DQN" $DQNWeight
Add-WeightArg $weightSpecs "PPO" $PPOWeight
Add-WeightArg $weightSpecs "RA-DDQN" $RADDQNWeight
Add-WeightArg $weightSpecs "PB-SafeDQN" $PBSafeDQNWeight
Add-WeightArg $weightSpecs "COBRA-Oracle" $COBRAWeight

if (-not [string]::IsNullOrWhiteSpace($HCRLWeightDir)) {
    Add-WeightArg $weightSpecs "HCRL-Oracle" $HCRLWeightDir
}
Add-WeightArg $weightSpecs "HCRL_Mode" $HCRLModeWeight
Add-WeightArg $weightSpecs "HCRL_Primary" $HCRLPrimaryWeight
Add-WeightArg $weightSpecs "HCRL_Backup" $HCRLBackupWeight

$loadArgs = @()
if ($weightSpecs.Count -gt 0) {
    $loadArgs += "--Load_Weights"
    $loadArgs += [string[]]$weightSpecs
} else {
    Write-Warning "No pretrained weight path was provided. The script will still run, but policies will use randomly initialized weights."
}

foreach ($profile in $AttackProfiles) {
    foreach ($intensity in $AttackIntensities) {
        foreach ($mr in $MaliciousRatios) {
            foreach ($s in $Seeds) {
                $pct = [int][Math]::Round($mr * 100)
                $tag = "attack_${profile}_I${intensity}_mal${pct}pct_eval_seed${s}"
                Write-Host "=== Attack profile=$profile intensity=$intensity malicious=$pct% seed=$s ==="

                $argsList = @(
                    "main.py",
                    "--Methods"
                ) + $Methods + @(
                    "--Scenario", $Scenario,
                    "--Use_Audit_Reputation",
                    "--Eval_Only",
                    "--Greedy_Eval",
                    "--Epoch", $Epoch,
                    "--Request_Num", $RequestNum,
                    "--Oracles_Per_Type", $OraclesPerType,
                    "--Malicious_Ratio", $mr,
                    "--Malicious_Placement", "balanced",
                    "--Attack_Profile", $profile,
                    "--Attack_Intensity", $intensity,
                    "--Reward_Scale", "3.0",
                    "--Reward_Clip", "3.0",
                    "--HCRL_lr", "0.0008",
                    "--HCRL_Mode_lr", "0.000616",
                    "--Output_Dir", $OutputDir,
                    "--Run_Tag", $tag,
                    "--Seed", $s
                ) + $loadArgs

                & python @argsList
                if ($LASTEXITCODE -ne 0) {
                    throw "Experiment failed for profile=$profile malicious=$pct seed=$s"
                }
            }
        }
    }
}
