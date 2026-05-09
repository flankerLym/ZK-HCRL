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
import sys
import atexit
import json
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from env import SchedulingEnv
from model import baseline_DQN, baseline_PPO, DuelingDoubleDQN, baselines, BLOR
from utils import get_args


# -----------------------------------------------------------------------------
# Automatic run logging and run folder
# -----------------------------------------------------------------------------
# Each run creates one self-contained folder under the CURRENT WORKING DIRECTORY by default:
#   <cwd>/output/YY_M_D_HH_MM_Epoch{N}_Req{M}_{Scenario}/
# containing:
#   - full console log .txt
#   - final results .csv/.json
#   - DQN / PPO / RA-DDQN weight checkpoints .npz
#   - reward curve pdf
# NOTE: do not create output folder here. We parse args first, then resolve
# args.Output_Dir from Path.cwd() so results are saved in the repository folder
# where the user runs `python main.py`, not in a downloaded replacement folder.


class Tee:
    """Write console output to both terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass
        self.flush()

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


# Capture get_args() prints first, because the run folder name needs args.Epoch
# and args.Request_Num.
_pre_arg_stdout = sys.stdout
_pre_arg_stderr = sys.stderr
_pre_arg_buffer = io.StringIO()
sys.stdout = _pre_arg_buffer
sys.stderr = _pre_arg_buffer
args = get_args()
sys.stdout = _pre_arg_stdout
sys.stderr = _pre_arg_stderr

_run_now = datetime.now()
RUN_ID = (
    f"{_run_now.year % 100}_{_run_now.month}_{_run_now.day}_"
    f"{_run_now.hour:02d}_{_run_now.minute:02d}_"
    f"Epoch{args.Epoch}_Req{args.Request_Num}_{args.Scenario}"
)
# Resolve output directory from the command working directory.
# This fixes cases where files were saved into a downloaded zip/replacement path.
_output_arg = Path(getattr(args, "Output_Dir", "output"))
if _output_arg.is_absolute():
    OUTPUT_BASE = _output_arg
else:
    OUTPUT_BASE = Path.cwd() / _output_arg
OUTPUT_BASE = OUTPUT_BASE.resolve()
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

RUN_DIR = (OUTPUT_BASE / RUN_ID).resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)
RUN_TXT_PATH = str(RUN_DIR / f"{RUN_ID}.txt")
RUN_CSV_PATH = str(RUN_DIR / f"{RUN_ID}_final_results.csv")
RUN_JSON_PATH = str(RUN_DIR / f"{RUN_ID}_final_results.json")

_log_f = open(RUN_TXT_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_f)
sys.stderr = Tee(sys.__stderr__, _log_f)
atexit.register(_log_f.close)

print(f"Current working directory: {Path.cwd().resolve()}")
print(f"Output base directory: {OUTPUT_BASE}")
print(f"Run folder: {RUN_DIR}")
print(f"Run log path: {RUN_TXT_PATH}")
# Replay argument/config prints into both terminal and log.
print(_pre_arg_buffer.getvalue(), end="")

np.random.seed(args.Seed)
random.seed(args.Seed)

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
# PB-SafeDQN diagnostics. Non-PB methods will stay at zero.
performance_primary_success = np.zeros(args.Baseline_num)
performance_backup_used = np.zeros(args.Baseline_num)
performance_backup_recovery = np.zeros(args.Baseline_num)
performance_backup_skipped = np.zeros(args.Baseline_num)
performance_conditional_backup_recovery = np.zeros(args.Baseline_num)
performance_backup_score_mean = np.zeros(args.Baseline_num)
performance_primary_malicious_num = np.zeros(args.Baseline_num)
performance_backup_malicious_num = np.zeros(args.Baseline_num)
performance_primary_trusted_num = np.zeros(args.Baseline_num)
performance_backup_trusted_num = np.zeros(args.Baseline_num)

# Generate environment.
env = SchedulingEnv(args)

# Build models.
brainOthers = baselines(env.actionNum, env.oracleTypes)
brainDQN = None
brainPPO = None
brainRA = None
brainPB = None
brainCOBRA = None
cobra_warm_started = False

if "DQN" in args.Baselines:
    brainDQN = baseline_DQN(
        env.actionNum, env.s_features,
        hidden_units=args.Dqn_hidden,
        scope="DQN",
        learning_rate=args.Dqn_lr,
        memory_size=args.Dqn_memory_size,
        batch_size=args.Dqn_batch_size,
        e_greedy_increment=args.Dqn_epsilon_increment,
        reward_clip=args.Reward_Clip,
        seed=args.Seed,
    )

if "PPO" in args.Baselines:
    brainPPO = baseline_PPO(
        env.actionNum,
        env.s_features,
        batch_size=args.PPO_batch_size,
        update_epochs=args.PPO_update_epochs,
        hidden_units=args.PPO_hidden,
        scope="PPO",
        seed=args.Seed,
    )

if "RA-DDQN" in args.Baselines:
    brainRA = DuelingDoubleDQN(
        env.actionNum, env.s_features,
        hidden_units=args.Dqn_hidden,
        scope="RA_DDQN",
        learning_rate=args.RA_lr,
        memory_size=args.Dqn_memory_size,
        batch_size=args.Dqn_batch_size,
        e_greedy_increment=args.Dqn_epsilon_increment,
        reward_clip=args.Reward_Clip,
        seed=args.Seed,
    )

if "PB-SafeDQN" in args.Baselines:
    brainPB = DuelingDoubleDQN(
        env.actionNum, env.s_features,
        hidden_units=args.Dqn_hidden,
        scope="PB_SafeDQN",
        learning_rate=args.PB_lr,
        memory_size=args.Dqn_memory_size,
        batch_size=args.Dqn_batch_size,
        e_greedy_increment=args.Dqn_epsilon_increment,
        reward_clip=args.Reward_Clip,
        seed=args.Seed + 1009,
    )

if "COBRA-Oracle" in args.Baselines:
    brainCOBRA = DuelingDoubleDQN(
        env.actionNum, env.s_features,
        hidden_units=args.Dqn_hidden,
        scope="COBRA_Oracle",
        learning_rate=args.COBRA_lr,
        memory_size=args.Dqn_memory_size,
        batch_size=args.Dqn_batch_size,
        e_greedy_increment=args.Dqn_epsilon_increment,
        reward_clip=args.Reward_Clip,
        seed=args.Seed + 2027,
    )

global_step = 0
# Use one consistent evaluation window for per-episode and final summaries.
# This excludes the warm-up/exploration prefix when Request_Num is small.
eval_start = min(2000, max(0, args.Request_Num // 2))

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
    has_last_pb = False
    has_last_cobra = False

    if brainDQN is not None:
        brainDQN.reward_list.clear()
    if brainPPO is not None:
        brainPPO.reward_list.clear()
    if brainRA is not None:
        brainRA.reward_list.clear()
    if brainPB is not None:
        brainPB.reward_list.clear()
    if brainCOBRA is not None:
        brainCOBRA.reward_list.clear()

    # Teacher-guided warm-start for COBRA. This copies the current teacher
    # representation into COBRA once, after the teacher has had several
    # episodes to learn a strong single-oracle primary policy.
    if brainCOBRA is not None and (not cobra_warm_started) and (not args.COBRA_No_Teacher) and episode >= args.COBRA_WarmStart_Episode:
        teacher_model = None
        if args.COBRA_Teacher_Source == "DQN":
            teacher_model = brainDQN
        elif args.COBRA_Teacher_Source == "RA-DDQN":
            teacher_model = brainRA
        if teacher_model is not None and brainCOBRA.copy_from(teacher_model, copy_optimizer_state=False):
            brainCOBRA.set_epsilon(min(getattr(teacher_model, "epsilon", 0.2), 0.35))
            cobra_warm_started = True
            print(f"[COBRA] warm-started primary selector from {args.COBRA_Teacher_Source} at episode {episode}")

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
            dqn_mask = env.get_action_mask(request_attrs)
            if has_last_dqn:
                brainDQN.store_transition(last_dqn_state, last_dqn_action, last_dqn_reward, dqn_state, dqn_mask)
            action_DQN = brainDQN.choose_action(dqn_state, dqn_mask)
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
                brainPPO.store_transition(last_ppo_state, last_ppo_action, last_ppo_reward, last_ppo_prob, last_ppo_mask)
            ppo_mask = env.get_action_mask(request_attrs)
            action_PPO, prob_PPO = brainPPO.choose_action(ppo_state, ppo_mask)
            reward_PPO = env.feedback(request_attrs, action_PPO, "PPO")
            if (global_step > args.PPO_start_learn) and (global_step % args.PPO_learn_interval == 0):
                brainPPO.learn()
            last_ppo_state = ppo_state
            last_ppo_action = action_PPO
            last_ppo_reward = reward_PPO
            last_ppo_prob = prob_PPO
            last_ppo_mask = ppo_mask
            has_last_ppo = True

        # Optional RA-DDQN policy.
        if brainRA is not None:
            ra_state = env.getState(request_attrs, "RA-DDQN")
            ra_mask = env.get_action_mask(request_attrs)
            if has_last_ra:
                brainRA.store_transition(last_ra_state, last_ra_action, last_ra_reward, ra_state, ra_mask)
            action_RA = brainRA.choose_action(ra_state, ra_mask)
            reward_RA = env.feedback(request_attrs, action_RA, "RA-DDQN")
            if (global_step > args.RA_start_learn) and (global_step % args.RA_learn_interval == 0):
                brainRA.learn()
            last_ra_state = ra_state
            last_ra_action = action_RA
            last_ra_reward = reward_RA
            has_last_ra = True

        # Optional PB-SafeDQN policy: Dueling Double DQN primary selector +
        # reputation/load/cost-aware backup failover.
        if brainPB is not None:
            pb_state = env.getState(request_attrs, "PB-SafeDQN")
            pb_mask = env.get_action_mask(request_attrs)
            if has_last_pb:
                brainPB.store_transition(last_pb_state, last_pb_action, last_pb_reward, pb_state, pb_mask)
            action_PB = brainPB.choose_action(pb_state, pb_mask)
            reward_PB = env.feedback_primary_backup(request_attrs, action_PB, "PB-SafeDQN")
            if (global_step > args.PB_start_learn) and (global_step % args.PB_learn_interval == 0):
                brainPB.learn()
            last_pb_state = pb_state
            last_pb_action = action_PB
            last_pb_reward = reward_PB
            has_last_pb = True

        # COBRA-Oracle: teacher-guided primary + adaptive constrained backup.
        if brainCOBRA is not None:
            cobra_state = env.getState(request_attrs, "COBRA-Oracle")
            cobra_mask = env.get_action_mask(request_attrs)
            if has_last_cobra:
                brainCOBRA.store_transition(last_cobra_state, last_cobra_action, last_cobra_reward, cobra_state, cobra_mask)

            teacher_action = None
            if (not args.COBRA_No_Teacher) and episode < args.COBRA_Teacher_Guidance_Episodes:
                teacher_model = brainDQN if args.COBRA_Teacher_Source == "DQN" else (brainRA if args.COBRA_Teacher_Source == "RA-DDQN" else None)
                if teacher_model is not None:
                    teacher_action = teacher_model.choose_best_action(cobra_state, cobra_mask)
            if teacher_action is not None:
                frac = max(0.0, 1.0 - episode / max(args.COBRA_Teacher_Guidance_Episodes, 1))
                teacher_prob = max(args.COBRA_Min_Teacher_Prob, args.COBRA_Teacher_Start_Prob * frac)
            else:
                teacher_prob = 0.0

            if teacher_action is not None and np.random.rand() < teacher_prob:
                action_COBRA = int(teacher_action)
            else:
                action_COBRA = brainCOBRA.choose_action(cobra_state, cobra_mask)
            reward_COBRA = env.feedback_primary_backup(request_attrs, action_COBRA, "COBRA-Oracle")
            if (global_step > args.COBRA_start_learn) and (global_step % args.COBRA_learn_interval == 0):
                brainCOBRA.learn()
            last_cobra_state = cobra_state
            last_cobra_action = action_COBRA
            last_cobra_reward = reward_COBRA
            has_last_cobra = True

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

    # Use the same evaluation window as the final summary.
    startP = eval_start

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
    for diag_name in ["PB-SafeDQN", "COBRA-Oracle"]:
        if diag_name in args.Baselines:
            idx_diag = args.Baselines.index(diag_name)
            print(
                f'[{diag_name} diagnostics]',
                'primary_success_rate:', env.get_totalPrimarySuccessRate(args.Baseline_num, startP)[idx_diag],
                'backup_used_rate:', env.get_totalBackupUsedRate(args.Baseline_num, startP)[idx_diag],
                'backup_recovery_rate:', env.get_totalBackupRecoveryRate(args.Baseline_num, startP)[idx_diag],
                'conditional_backup_recovery_rate:', env.get_totalConditionalBackupRecoveryRate(args.Baseline_num, startP)[idx_diag],
                'backup_skipped_rate:', env.get_totalBackupSkippedRate(args.Baseline_num, startP)[idx_diag],
                'backup_score_mean:', env.get_totalBackupScoreMean(args.Baseline_num, startP)[idx_diag],
            )

    if episode != 0 or args.Epoch == 1:
        performance_lamda += env.get_total_responseTs(args.Baseline_num, eval_start)
        performance_total_rewards += env.get_totalRewards(args.Baseline_num, eval_start)
        performance_success += env.get_totalSuccess(args.Baseline_num, eval_start)
        performance_success_time += env.get_totalSuccessInTime(args.Baseline_num, eval_start)
        performance_finishT += env.get_totalTimes(args.Baseline_num, eval_start)
        performance_cost += env.get_totalCost(args.Baseline_num, eval_start)
        performance_match += env.get_totalMatchRate(args.Baseline_num, eval_start)
        performance_assigned_malicious_num += env.get_totalMaliciousNum(args.Baseline_num, eval_start)
        performance_assigned_normal_num += env.get_totalNormalNum(args.Baseline_num, eval_start)
        performance_assigned_trusted_num += env.get_totalTrustedNum(args.Baseline_num, eval_start)
        performance_primary_success += env.get_totalPrimarySuccessRate(args.Baseline_num, eval_start)
        performance_backup_used += env.get_totalBackupUsedRate(args.Baseline_num, eval_start)
        performance_backup_recovery += env.get_totalBackupRecoveryRate(args.Baseline_num, eval_start)
        performance_backup_skipped += env.get_totalBackupSkippedRate(args.Baseline_num, eval_start)
        performance_conditional_backup_recovery += env.get_totalConditionalBackupRecoveryRate(args.Baseline_num, eval_start)
        performance_backup_score_mean += env.get_totalBackupScoreMean(args.Baseline_num, eval_start)
        performance_primary_malicious_num += env.get_totalPrimaryMaliciousNum(args.Baseline_num, eval_start)
        performance_backup_malicious_num += env.get_totalBackupMaliciousNum(args.Baseline_num, eval_start)
        performance_primary_trusted_num += env.get_totalPrimaryTrustedNum(args.Baseline_num, eval_start)
        performance_backup_trusted_num += env.get_totalBackupTrustedNum(args.Baseline_num, eval_start)

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
        plt.savefig(os.path.join(RUN_DIR, f'{RUN_ID}_reward_episode{episode}.pdf'), dpi=600)
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
performance_primary_success = np.around(performance_primary_success / divisor, 3)
performance_backup_used = np.around(performance_backup_used / divisor, 3)
performance_backup_recovery = np.around(performance_backup_recovery / divisor, 3)
performance_backup_skipped = np.around(performance_backup_skipped / divisor, 3)
performance_conditional_backup_recovery = np.around(performance_conditional_backup_recovery / divisor, 3)
performance_backup_score_mean = np.around(performance_backup_score_mean / divisor, 3)
performance_primary_malicious_num = np.around(performance_primary_malicious_num / divisor, 0)
performance_backup_malicious_num = np.around(performance_backup_malicious_num / divisor, 0)
performance_primary_trusted_num = np.around(performance_primary_trusted_num / divisor, 0)
performance_backup_trusted_num = np.around(performance_backup_trusted_num / divisor, 0)

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
print('primary_success_rate:')
print(performance_primary_success)
print('backup_used_rate:')
print(performance_backup_used)
print('backup_recovery_rate:')
print(performance_backup_recovery)
print('conditional_backup_recovery_rate:')
print(performance_conditional_backup_recovery)
print('backup_skipped_rate:')
print(performance_backup_skipped)
print('backup_score_mean:')
print(performance_backup_score_mean)
print('primary malicious oracle count:')
print(performance_primary_malicious_num)
print('backup malicious oracle count:')
print(performance_backup_malicious_num)
print('primary trusted oracle count:')
print(performance_primary_trusted_num)
print('backup trusted oracle count:')
print(performance_backup_trusted_num)

# Save machine-readable final results with the same timestamp prefix as the .txt log.
final_results_df = pd.DataFrame({
    "method": list(args.Baselines),
    "avg_responseT": performance_lamda,
    "total_rewards": performance_total_rewards,
    "success_rate": performance_success,
    "success_time_rate": performance_success_time,
    "finishT": performance_finishT,
    "cost": performance_cost,
    "match_rate": performance_match,
    "assigned_malicious": performance_assigned_malicious_num,
    "assigned_normal": performance_assigned_normal_num,
    "assigned_trusted": performance_assigned_trusted_num,
    "primary_success_rate": performance_primary_success,
    "backup_used_rate": performance_backup_used,
    "backup_recovery_rate": performance_backup_recovery,
    "conditional_backup_recovery_rate": performance_conditional_backup_recovery,
    "backup_skipped_rate": performance_backup_skipped,
    "backup_score_mean": performance_backup_score_mean,
    "primary_malicious": performance_primary_malicious_num,
    "backup_malicious": performance_backup_malicious_num,
    "primary_trusted": performance_primary_trusted_num,
    "backup_trusted": performance_backup_trusted_num,
})
final_results_df.to_csv(RUN_CSV_PATH, index=False, encoding="utf-8-sig")

# Save model weights into the same run folder as the log/results.
# Filenames include timestamp, training epochs, request count and scenario via RUN_ID.
weight_paths = {}
weight_metadata = {
    "run_id": RUN_ID,
    "epoch": int(args.Epoch),
    "request_num": int(args.Request_Num),
    "scenario": args.Scenario,
    "success_mode": args.Success_Mode,
    "state_mode": args.State_Mode,
    "reward_mode": args.Reward_Mode,
    "action_mask_mode": args.Action_Mask_Mode,
    "seed": int(args.Seed),
    "baselines": list(args.Baselines),
    "eval_start": int(eval_start),
}
if brainDQN is not None:
    weight_paths["DQN"] = brainDQN.save_model(
        os.path.join(RUN_DIR, f"{RUN_ID}_DQN_weights.npz"),
        metadata={**weight_metadata, "method": "DQN"},
    )
if brainPPO is not None:
    weight_paths["PPO"] = brainPPO.save_model(
        os.path.join(RUN_DIR, f"{RUN_ID}_PPO_weights.npz"),
        metadata={**weight_metadata, "method": "PPO"},
    )
if brainRA is not None:
    weight_paths["RA-DDQN"] = brainRA.save_model(
        os.path.join(RUN_DIR, f"{RUN_ID}_RA-DDQN_weights.npz"),
        metadata={**weight_metadata, "method": "RA-DDQN"},
    )
if brainPB is not None:
    weight_paths["PB-SafeDQN"] = brainPB.save_model(
        os.path.join(RUN_DIR, f"{RUN_ID}_PB-SafeDQN_weights.npz"),
        metadata={**weight_metadata, "method": "PB-SafeDQN", "backup_mode": args.PB_Backup_Mode, "backup_trigger": args.PB_Backup_Trigger, "min_backup_score": args.PB_Min_Backup_Score},
    )
if brainCOBRA is not None:
    weight_paths["COBRA-Oracle"] = brainCOBRA.save_model(
        os.path.join(RUN_DIR, f"{RUN_ID}_COBRA-Oracle_weights.npz"),
        metadata={**weight_metadata, "method": "COBRA-Oracle", "teacher_source": args.COBRA_Teacher_Source, "gate_mode": args.COBRA_Gate_Mode, "min_backup_score": args.COBRA_Min_Backup_Score},
    )

print('saved model weights:')
if len(weight_paths) == 0:
    print('No DQN/PPO/RA-DDQN model was enabled, so no weight file was saved.')
else:
    for method_name, path in weight_paths.items():
        print(f"{method_name}: {path}")

run_metadata = {
    "run_id": RUN_ID,
    "run_dir": str(RUN_DIR),
    "output_base": str(OUTPUT_BASE),
    "log_txt": RUN_TXT_PATH,
    "final_results_csv": RUN_CSV_PATH,
    "weight_paths": weight_paths,
    "eval_start": int(eval_start),
    "epoch": int(args.Epoch),
    "request_num": int(args.Request_Num),
    "scenario": args.Scenario,
    "success_mode": args.Success_Mode,
    "state_mode": args.State_Mode,
    "reward_mode": args.Reward_Mode,
    "action_mask_mode": args.Action_Mask_Mode,
    "pb_backup_trigger": getattr(args, "PB_Backup_Trigger", None),
    "pb_min_backup_score": float(getattr(args, "PB_Min_Backup_Score", 0.0)),
    "cobra_gate_mode": getattr(args, "COBRA_Gate_Mode", None),
    "cobra_teacher_source": getattr(args, "COBRA_Teacher_Source", None),
    "cobra_min_backup_score": float(getattr(args, "COBRA_Min_Backup_Score", 0.0)),
    "baselines": list(args.Baselines),
    "seed": int(args.Seed),
    "reward_scale": float(args.Reward_Scale),
    "reward_clip": float(args.Reward_Clip),
}
with open(RUN_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": run_metadata,
        "final_results": final_results_df.to_dict(orient="records"),
    }, f, indent=2, ensure_ascii=False)

print('saved run log txt:')
print(RUN_TXT_PATH)
print('saved final results csv:')
print(RUN_CSV_PATH)
print('saved final results json:')
print(RUN_JSON_PATH)
