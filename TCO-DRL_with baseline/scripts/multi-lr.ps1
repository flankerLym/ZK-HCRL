$seed = 3
$lrs = @(0.0005, 0.0008, 0.0010, 0.0013, 0.0016, 0.0020)

foreach ($lr in $lrs) {
  $mode_lr = [Math]::Round($lr * 0.77, 6)
  $lr_tag = ("{0:F6}" -f $lr).Replace(".", "p")
  $mode_tag = ("{0:F6}" -f $mode_lr).Replace(".", "p")

  python main.py `
    --Methods DQN HCRL-Oracle `
    --Scenario rl_harder `
    --Use_Audit_Reputation `
    --Epoch 100 `
    --Request_Num 6000 `
    --Oracles_Per_Type 10 `
    --Reward_Scale 3.0 `
    --Reward_Clip 3.0 `
    --HCRL_lr $lr `
    --HCRL_Mode_lr $mode_lr `
    --Run_Tag "convergence_coupled_lr_${lr_tag}_mode_${mode_tag}_epoch100_seed_${seed}" `
    --Seed $seed
}