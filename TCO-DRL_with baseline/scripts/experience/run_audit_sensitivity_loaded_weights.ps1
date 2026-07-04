<#
Run audit-sensitivity tests with pretrained weights.

Usage example:
  .\scripts\run_audit_sensitivity_loaded_weights.ps1 `
    -HCRLWeightDir "output\your_train_run" `
    -DQNWeight "output\your_train_run\DQN.npz"
#>
param(
    [string[]]$Methods = @("DQN", "HCRL-Oracle"),
    [int[]]$Seeds = @(3),
    [double]$MaliciousRatio = 0.30,
    [int]$Epoch = 10,
    [int]$RequestNum = 6000,
    [int]$OraclesPerType = 40,
    [string]$Scenario = "rl_harder",
    [string]$AttackProfile = "mixed",
    [double]$AttackIntensity = 1.0,
    [string]$OutputDir = "output",

    # Weight paths.
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

# One-factor-style audit settings. The first setting is an audit-off ablation;
# the others gradually increase audit rate and reputation correction strength.
$auditSettings = @(
    @{ Name = "audit_off";  Base = 0.00; Risk = 0.00; Weight = 0.00; FailPenalty = 0.00; PassRecovery = 0.00; Disable = $true  },
    @{ Name = "audit_low";  Base = 0.01; Risk = 0.05; Weight = 0.15; FailPenalty = 0.04; PassRecovery = 0.02; Disable = $false },
    @{ Name = "audit_mid";  Base = 0.03; Risk = 0.10; Weight = 0.30; FailPenalty = 0.08; PassRecovery = 0.03; Disable = $false },
    @{ Name = "audit_high"; Base = 0.05; Risk = 0.15; Weight = 0.45; FailPenalty = 0.12; PassRecovery = 0.04; Disable = $false },
    @{ Name = "audit_vhigh";Base = 0.08; Risk = 0.20; Weight = 0.60; FailPenalty = 0.16; PassRecovery = 0.05; Disable = $false }
)

foreach ($setting in $auditSettings) {
    foreach ($s in $Seeds) {
        $pct = [int][Math]::Round($MaliciousRatio * 100)
        $tag = "audit_sens_$($setting.Name)_mal${pct}pct_eval_seed${s}"
        Write-Host "=== Audit setting=$($setting.Name) malicious=$pct% seed=$s ==="

        $auditArgs = @()
        if ($setting.Disable) {
            $auditArgs += "--Disable_Audit_Reputation"
        } else {
            $auditArgs += "--Use_Audit_Reputation"
        }

        $argsList = @(
            "main.py",
            "--Methods"
        ) + $Methods + @(
            "--Scenario", $Scenario,
            "--Eval_Only",
            "--Greedy_Eval",
            "--Epoch", $Epoch,
            "--Request_Num", $RequestNum,
            "--Oracles_Per_Type", $OraclesPerType,
            "--Malicious_Ratio", $MaliciousRatio,
            "--Malicious_Placement", "balanced",
            "--Attack_Profile", $AttackProfile,
            "--Attack_Intensity", $AttackIntensity,
            "--Audit_Base_Rate", $setting.Base,
            "--Audit_Risk_Rate", $setting.Risk,
            "--Audit_Weight_In_Reputation", $setting.Weight,
            "--Audit_Fail_Penalty", $setting.FailPenalty,
            "--Audit_Pass_Recovery", $setting.PassRecovery,
            "--Reward_Scale", "3.0",
            "--Reward_Clip", "3.0",
            "--HCRL_lr", "0.0008",
            "--HCRL_Mode_lr", "0.000616",
            "--Output_Dir", $OutputDir,
            "--Run_Tag", $tag,
            "--Seed", $s
        ) + $auditArgs + $loadArgs

        & python @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment failed for audit setting=$($setting.Name) seed=$s"
        }
    }
}
