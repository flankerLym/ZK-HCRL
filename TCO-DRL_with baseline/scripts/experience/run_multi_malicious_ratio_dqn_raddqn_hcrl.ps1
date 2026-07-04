$ErrorActionPreference = "Stop"

$seeds = @(3)
$oracle_nums = @(50,30,20,5)
$malicious_ratios = @(0.0, 0.1, 0.2, 0.3, 0.4)

foreach ($mr in $malicious_ratios) {
  $mr_pct = [int][Math]::Round($mr * 100)
  foreach ($o in $oracle_nums) {
    foreach ($s in $seeds) {
      python main.py `
        --Methods DQN RA-DDQN HCRL-Oracle `
        --Scenario rl_harder `
        --Use_Audit_Reputation `
        --Epoch 30 `
        --Request_Num 6000 `
        --Oracles_Per_Type $o `
        --Malicious_Ratio $mr `
        --Malicious_Placement balanced `
        --Reward_Scale 3.0 `
        --Reward_Clip 3.0 `
        --HCRL_lr 0.0008 `
        --HCRL_Mode_lr 0.000616 `
        --Run_Tag "oracle_scale_o${o}_mal${mr_pct}pct_lr8_hcrl_ra" `
        --Seed $s
    }
  }
}
