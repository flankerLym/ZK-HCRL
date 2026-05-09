# Quick smoke test. Run from repository root or from TCO-DRL_with baseline folder.
Set-Location "$PSScriptRoot\..\TCO-DRL_with baseline"
python main.py --Scenario rl_harder --Use_RA_DDQN --Use_PB_SafeDQN --Use_COBRA --Oracles_Per_Type 10 --Epoch 5 --Request_Num 1500 --Reward_Scale 2.0 --Reward_Clip 2.0 --Dqn_batch_size 64 --Dqn_memory_size 3000
