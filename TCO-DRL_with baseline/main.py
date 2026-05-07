import os
# Silence TensorFlow GPU / CUDA probing messages on CPU-only Windows machines.
# Must be set before importing tensorflow through model.py.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import warnings
warnings.filterwarnings("ignore", message="`tf.layers.dense` is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from env import SchedulingEnv
from model import baseline_DQN, baseline_PPO, DuelingDoubleDQN, baselines, BLOR
from utils import get_args


args = get_args()
np.random.seed(args.Seed)
random.seed(args.Seed)

os.makedirs("./output", exist_ok=True)

# Store final averaged results.
performance_lamda = np.zeros(args.Baseline_num)
performance_total_rewards = np.zeros(args.Baseline_num)
performance_success = np.zeros(args.Baseline_num)
performance_success_time = np.zeros(args.Baseline_num)
performance_match = np.zeros(args.Baseline_num)
performance_finishT = np.zeros(args.Baseline_num)
performance_cost = np.zeros(args.Baseline_num)
performance_assigned_malicious_num = np.zeros(args.Baseline_num)
performance_assigned_normal_num = np.zeros(args.Baseline_num)
performance_assigned_trusted_num = np.zeros(args.Baseline_num)

# Generate environment.
env = SchedulingEnv(args)

# Build models.
brainOthers = baselines(env.actionNum, env.oracleTypes)
brainDQN = None
brainPPO = None
brainRA = None

if "DQN" in args.Baselines:
    brainDQN = baseline_DQN(env.actionNum, env.s_features, hidden_units=args.Dqn_hidden, scope="DQN")

if "PPO" in args.Baselines:
    brainPPO = baseline_PPO(
        env.actionNum,
        env.s_features,
        batch_size=args.PPO_batch_size,
        update_epochs=args.PPO_update_epochs,
        hidden_units=args.PPO_hidden,
        scope="PPO",
    )

if "RA-DDQN" in args.Baselines:
    brainRA = DuelingDoubleDQN(env.actionNum, env.s_features, hidden_units=args.Dqn_hidden, scope="RA_DDQN")

global_step = 0

for episode in range(args.Epoch):
    print('----------------------------Episode', episode, '----------------------------')
    request_c = 1
    time_period = 1
    BLOR_c = 1
    performance_c = 0

    env.reset(args)
    env.reset_reputation_factors()
    env.initial_reputation()

    has_last_dqn = False
    has_last_ppo = False
    has_last_ra = False

    if brainDQN is not None:
        brainDQN.reward_list.clear()
    if brainPPO is not None:
        brainPPO.reward_list.clear()
    if brainRA is not None:
        brainRA.reward_list.clear()

    while True:
        # Update reputation every Time_Period_Size requests.
        if request_c % args.Time_Period_Size == 0:
            time_period += 1
            for policy_name in args.Baselines:
                reputation_attributes = env.get_reputation_factors(policy_name)
                env.update_reputation(reputation_attributes, time_period, policy_name)
            env.reset_reputation_factors()

        global_step += 1
        finish, request_attrs = env.workload(request_c)

        # Random policy.
        if "Random" in args.Baselines:
            action_random = brainOthers.random_choose_action()
            env.feedback(request_attrs, action_random, "Random")

        # Round-Robin policy.
        if "Round-Robin" in args.Baselines:
            action_RR = brainOthers.RR_choose_action(request_c)
            env.feedback(request_attrs, action_RR, "Round-Robin")

        # Earliest policy.
        if "Earliest" in args.Baselines:
            idle_times = env.get_oracle_idleT("Earliest")
            action_early = brainOthers.early_choose_action(idle_times)
            env.feedback(request_attrs, action_early, "Earliest")

        # DQN / original TCO-DRL.
        if brainDQN is not None:
            dqn_state = env.getState(request_attrs, "DQN")
            if has_last_dqn:
                brainDQN.store_transition(last_dqn_state, last_dqn_action, last_dqn_reward, dqn_state)
            action_DQN = brainDQN.choose_action(dqn_state)
            reward_DQN = env.feedback(request_attrs, action_DQN, "DQN")
            if (global_step > args.Dqn_start_learn) and (global_step % args.Dqn_learn_interval == 0):
                brainDQN.learn()
            last_dqn_state = dqn_state
            last_dqn_action = action_DQN
            last_dqn_reward = reward_DQN
            has_last_dqn = True

        # BLOR policy.
        if "BLOR" in args.Baselines:
            start_counter = (BLOR_c - 1) * 200
            RR_counter = start_counter + min(15, env.actionNum)
            end_counter = BLOR_c * 200
            if request_c > start_counter and request_c < RR_counter + 1:
                action_BLOR = brainOthers.RR_choose_action(request_c)
            elif request_c > RR_counter and request_c < end_counter + 1:
                request_num_BLOR = env.get_request_num("BLOR")
                success_num_BLOR = env.get_successful_validation("BLOR")
                failure_num_BLOR = np.maximum(request_num_BLOR - success_num_BLOR, 0)
                brainBLOR = BLOR(success_num_BLOR, failure_num_BLOR, env.oracleCost)
                oracles_BLOR = brainBLOR.get_oracles(success_num_BLOR, failure_num_BLOR, env.oracleCost)
                action_BLOR = brainBLOR.choose_action(oracles_BLOR)
            else:
                action_BLOR = brainOthers.RR_choose_action(request_c)
            env.feedback(request_attrs, action_BLOR, "BLOR")
            if request_c % 200 == 0:
                env.reset_reputation_factors_BLOR()
                BLOR_c += 1

        # SemiGreedy policy.
        if "SemiGreedy" in args.Baselines:
            rewards_PSG, cost_PSG = env.feedback_PSG_FWA(request_attrs, "SemiGreedy")
            action_PSG = brainOthers.PSG_choose_action(rewards_PSG, cost_PSG)
            env.feedback(request_attrs, action_PSG, "SemiGreedy")

        # PPO policy.
        if brainPPO is not None:
            ppo_state = env.getState(request_attrs, "PPO")
            if has_last_ppo:
                brainPPO.store_transition(last_ppo_state, last_ppo_action, last_ppo_reward, last_ppo_prob)
            action_PPO, prob_PPO = brainPPO.choose_action(ppo_state)
            reward_PPO = env.feedback(request_attrs, action_PPO, "PPO")
            if (global_step > args.PPO_start_learn) and (global_step % args.PPO_learn_interval == 0):
                brainPPO.learn()
            last_ppo_state = ppo_state
            last_ppo_action = action_PPO
            last_ppo_reward = reward_PPO
            last_ppo_prob = prob_PPO
            has_last_ppo = True

        # Optional RA-DDQN policy.
        if brainRA is not None:
            ra_state = env.getState(request_attrs, "RA-DDQN")
            if has_last_ra:
                brainRA.store_transition(last_ra_state, last_ra_action, last_ra_reward, ra_state)
            action_RA = brainRA.choose_action(ra_state)
            reward_RA = env.feedback(request_attrs, action_RA, "RA-DDQN")
            if (global_step > args.RA_start_learn) and (global_step % args.RA_learn_interval == 0):
                brainRA.learn()
            last_ra_state = ra_state
            last_ra_action = action_RA
            last_ra_reward = reward_RA
            has_last_ra = True

        if request_c % 500 == 0:
            # Keep these calls for compatibility with the original code, even if the values are not printed here.
            env.get_accumulateRewards(args.Baseline_num, performance_c, request_c)
            env.get_accumulateCost(args.Baseline_num, performance_c, request_c)
            env.get_FinishTimes(args.Baseline_num, performance_c, request_c)
            env.get_executeTs(args.Baseline_num, performance_c, request_c)
            env.get_waitTs(args.Baseline_num, performance_c, request_c)
            env.get_responseTs(args.Baseline_num, performance_c, request_c)
            env.get_successTimes(args.Baseline_num, performance_c, request_c)
            env.get_successInTime(args.Baseline_num, performance_c, request_c)
            performance_c = request_c

        request_c += 1
        if finish:
            break

    # Avoid out-of-bounds when Request_Num <= 2000.
    startP = min(2000, max(0, args.Request_Num // 2))

    total_Rewards = env.get_totalRewards(args.Baseline_num, startP)
    avg_allRespTs = env.get_total_responseTs(args.Baseline_num, startP)
    total_success = env.get_totalSuccess(args.Baseline_num, startP)
    total_success_time = env.get_totalSuccessInTime(args.Baseline_num, startP)
    total_Ts = env.get_totalTimes(args.Baseline_num, startP)
    total_cost = env.get_totalCost(args.Baseline_num, startP)

    print(f'total performance (after {startP} requests):')
    for i, name in enumerate(args.Baselines):
        print(
            f"[{name}] reward:", total_Rewards[i],
            ' avg_responseT:', avg_allRespTs[i],
            'success_rate:', total_success[i],
            'success_time_rate:', total_success_time[i],
            ' finishT:', total_Ts[i],
            'Cost:', total_cost[i],
        )

    if episode != 0:
        performance_lamda += env.get_total_responseTs(args.Baseline_num, 0)
        performance_total_rewards += env.get_totalRewards(args.Baseline_num, 0)
        performance_success += env.get_totalSuccess(args.Baseline_num, 0)
        performance_success_time += env.get_totalSuccessInTime(args.Baseline_num, 0)
        performance_finishT += env.get_totalTimes(args.Baseline_num, 0)
        performance_cost += env.get_totalCost(args.Baseline_num, 0)
        performance_match += env.get_totalMatchRate(args.Baseline_num)
        performance_assigned_malicious_num += env.get_totalMaliciousNum(args.Baseline_num)
        performance_assigned_normal_num += env.get_totalNormalNum(args.Baseline_num)
        performance_assigned_trusted_num += env.get_totalTrustedNum(args.Baseline_num)

    if episode == 0 and brainDQN is not None and len(brainDQN.reward_list) > 0:
        sns.set_style("darkgrid")
        window_size = 30
        rewards_series = pd.Series(brainDQN.reward_list)
        moving_avg_rewards = rewards_series.rolling(window=window_size).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(brainDQN.reward_list, label='DQN reward')
        plt.plot(moving_avg_rewards, label='moving average reward', color='darkorange')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'./output/reward_episode{episode}.pdf', dpi=600)
        plt.close()

print('---------------------------- Final results ----------------------------')
divisor = max(args.Epoch - 1, 1)
performance_lamda = np.around(performance_lamda / divisor, 3)
performance_total_rewards = np.around(performance_total_rewards / divisor, 3)
performance_success = np.around(performance_success / divisor, 3)
performance_success_time = np.around(performance_success_time / divisor, 3)
performance_finishT = np.around(performance_finishT / divisor, 3)
performance_cost = np.around(performance_cost / divisor, 5)
performance_match = np.around(performance_match / divisor, 3)
performance_assigned_malicious_num = np.around(performance_assigned_malicious_num / divisor, 0)
performance_assigned_normal_num = np.around(performance_assigned_normal_num / divisor, 0)
performance_assigned_trusted_num = np.around(performance_assigned_trusted_num / divisor, 0)

print('method order:')
print(args.Baselines)
print('avg_responseT:')
print(performance_lamda)
print('total_rewards:')
print(performance_total_rewards)
print('success_rate:')
print(performance_success)
print('success_time_rate:')
print(performance_success_time)
print('finishT:')
print(performance_finishT)
print('cost:')
print(performance_cost)
print('match rate:')
print(performance_match)
print('requests assigned to malicious oracle:')
print(performance_assigned_malicious_num)
print('requests assigned to normal oracle:')
print(performance_assigned_normal_num)
print('requests assigned to trusted oracle:')
print(performance_assigned_trusted_num)
