import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import io
import json
import atexit
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from env import SchedulingEnv
from model import baseline_DQN, baseline_PPO, DuelingDoubleDQN, OptionActorCritic, baselines, BLOR
from utils import get_args


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


_pre_stdout, _pre_stderr = sys.stdout, sys.stderr
_arg_buf = io.StringIO()
sys.stdout = _arg_buf
sys.stderr = _arg_buf
args = get_args()
sys.stdout, sys.stderr = _pre_stdout, _pre_stderr

np.random.seed(args.Seed)
random.seed(args.Seed)

now = datetime.now()
tag = str(getattr(args, "Run_Tag", "")).strip()
tag_part = f"_{tag}" if tag else ""
RUN_ID = f"{now.year%100}_{now.month}_{now.day}_{now.hour:02d}_{now.minute:02d}_Epoch{args.Epoch}_Req{args.Request_Num}_{args.Scenario}_Seed{args.Seed}{tag_part}"
out_base = Path(args.Output_Dir)
if not out_base.is_absolute():
    out_base = Path.cwd() / out_base
out_base.mkdir(parents=True, exist_ok=True)
RUN_DIR = (out_base / RUN_ID).resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)
RUN_TXT_PATH = RUN_DIR / f"{RUN_ID}.txt"
RUN_CSV_PATH = RUN_DIR / f"{RUN_ID}_final_results.csv"
RUN_JSON_PATH = RUN_DIR / f"{RUN_ID}_final_results.json"

_log_f = open(RUN_TXT_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_f)
sys.stderr = Tee(sys.__stderr__, _log_f)
atexit.register(_log_f.close)

print(f"Current working directory: {Path.cwd().resolve()}")
print(f"Run folder: {RUN_DIR}")
print(f"Run log path: {RUN_TXT_PATH}")
print(_arg_buf.getvalue(), end="")


def build_models(env, args):
    brain = {"others": baselines(env.actionNum, env.oracleTypes)}
    if "DQN" in args.Baselines:
        brain["DQN"] = baseline_DQN(env.actionNum, env.s_features, hidden_units=args.Dqn_hidden,
                                    scope="DQN", learning_rate=args.Dqn_lr,
                                    memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
                                    e_greedy_increment=args.Dqn_epsilon_increment,
                                    reward_clip=args.Reward_Clip, seed=args.Seed)
    if "PPO" in args.Baselines:
        brain["PPO"] = baseline_PPO(env.actionNum, env.s_features, batch_size=args.PPO_batch_size,
                                    update_epochs=args.PPO_update_epochs, hidden_units=args.PPO_hidden,
                                    scope="PPO", actor_lr=getattr(args, "PPO_lr", 0.0015), seed=args.Seed,
                                    reward_clip=args.Reward_Clip)
    if "RA-DDQN" in args.Baselines:
        brain["RA-DDQN"] = DuelingDoubleDQN(env.actionNum, env.s_features, hidden_units=args.Dqn_hidden,
                                            scope="RA_DDQN", learning_rate=args.RA_lr,
                                            memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
                                            e_greedy_increment=args.Dqn_epsilon_increment,
                                            reward_clip=args.Reward_Clip, seed=args.Seed + 17)
    if "PB-SafeDQN" in args.Baselines:
        brain["PB-SafeDQN"] = DuelingDoubleDQN(env.actionNum, env.s_features, hidden_units=args.Dqn_hidden,
                                               scope="PB_SafeDQN", learning_rate=args.PB_lr,
                                               memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
                                               e_greedy_increment=args.Dqn_epsilon_increment,
                                               reward_clip=args.Reward_Clip, seed=args.Seed + 1009)
    if "COBRA-Oracle" in args.Baselines:
        brain["COBRA-Oracle"] = DuelingDoubleDQN(env.actionNum, env.s_features, hidden_units=args.Dqn_hidden,
                                                 scope="COBRA_Oracle", learning_rate=args.COBRA_lr,
                                                 memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
                                                 e_greedy_increment=args.Dqn_epsilon_increment,
                                                 reward_clip=args.Reward_Clip, seed=args.Seed + 2027)
    if "HCRL-Oracle" in args.Baselines:
        mode_dim = env.s_features + env.mode_extra_features
        n_modes = len(args.HCRL_Mode_Names)
        brain["HCRL_Mode"] = OptionActorCritic(n_modes, mode_dim, hidden_units=max(64, args.Dqn_hidden // 2),
                                                scope="HCRL_Mode_OptionAC", learning_rate=args.HCRL_Mode_lr,
                                                memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
                                                entropy_coef=args.HCRL_AC_Entropy, value_coef=args.HCRL_AC_Value_Coef,
                                                reward_clip=args.Reward_Clip, seed=args.Seed + 3031)
        brain["HCRL_Primary"] = OptionActorCritic(env.actionNum, env.s_features, hidden_units=args.Dqn_hidden,
                                                   scope="HCRL_Primary_OptionAC", learning_rate=args.HCRL_lr,
                                                   memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
                                                   entropy_coef=args.HCRL_AC_Entropy, value_coef=args.HCRL_AC_Value_Coef,
                                                   reward_clip=args.Reward_Clip, seed=args.Seed + 4049)
        brain["HCRL_Backup"] = OptionActorCritic(env.actionNum, env.s_features, hidden_units=args.Dqn_hidden,
                                                  scope="HCRL_Backup_OptionAC", learning_rate=args.HCRL_lr,
                                                  memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
                                                  entropy_coef=args.HCRL_AC_Entropy, value_coef=args.HCRL_AC_Value_Coef,
                                                  reward_clip=args.Reward_Clip, seed=args.Seed + 5051)
    return brain


def _canonical_weight_key(name):
    key = str(name).strip().strip('"').strip("'")
    aliases = {
        "dqn": "DQN", "tco-drl": "DQN", "tco_drl": "DQN",
        "ppo": "PPO",
        "ra": "RA-DDQN", "ra-ddqn": "RA-DDQN", "ra_ddqn": "RA-DDQN", "raddqn": "RA-DDQN",
        "pb": "PB-SafeDQN", "pb-safedqn": "PB-SafeDQN", "pb_safedqn": "PB-SafeDQN",
        "cobra": "COBRA-Oracle", "cobra-oracle": "COBRA-Oracle", "cobra_oracle": "COBRA-Oracle",
        "hcrl": "HCRL-Oracle", "hcrl-oracle": "HCRL-Oracle", "hcrl_oracle": "HCRL-Oracle",
        "hcrl-mode": "HCRL_Mode", "hcrl_mode": "HCRL_Mode", "hcrlmode": "HCRL_Mode",
        "hcrl-primary": "HCRL_Primary", "hcrl_primary": "HCRL_Primary", "hcrlprimary": "HCRL_Primary",
        "hcrl-backup": "HCRL_Backup", "hcrl_backup": "HCRL_Backup", "hcrlbackup": "HCRL_Backup",
    }
    if key in ["DQN", "PPO", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle", "HCRL_Mode", "HCRL_Primary", "HCRL_Backup"]:
        return key
    return aliases.get(key.lower(), key)


def _parse_weight_specs(args):
    specs = {}
    for item in list(getattr(args, "Load_Weights", []) or []):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --Load_Weights item: {item}. Expected METHOD=path")
        k, v = item.split("=", 1)
        specs[_canonical_weight_key(k)] = v.strip().strip('"').strip("'")
    shortcut = str(getattr(args, "Weight_Path", "") or "").strip()
    if shortcut:
        selected_rl = [m for m in getattr(args, "Baselines", []) if m in ["DQN", "PPO", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle"]]
        if len(selected_rl) == 1:
            specs[selected_rl[0]] = shortcut
        else:
            raise ValueError("--Weight_Path shortcut requires exactly one selected RL method. Use --Load_Weights METHOD=path instead.")
    return specs


def _candidate_checkpoint_files(path, names):
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    if not p.exists():
        return []
    out = []
    for name in names:
        c = p / name
        if c.exists():
            out.append(c)
    return out


def _load_one_model(model, path, name, args):
    if model is None or not hasattr(model, "load_model"):
        print(f"[Weight load warning] {name} has no load_model(); skipped {path}")
        return False
    ok = model.load_model(str(path), strict=bool(getattr(args, "Strict_Weight_Load", False)))
    if ok:
        print(f"[Weight load] loaded {name} <- {path}")
    return bool(ok)


def load_requested_weights(brain, args):
    specs = _parse_weight_specs(args)
    loaded = {}
    if not specs:
        return loaded
    for method, path in specs.items():
        if method == "HCRL-Oracle":
            # A HCRL run folder contains three separate checkpoints.
            mapping = {
                "HCRL_Mode": ["HCRL_Mode.npz", "HCRL_Mode_OptionAC.npz", "HCRL-Mode.npz"],
                "HCRL_Primary": ["HCRL_Primary.npz", "HCRL_Primary_OptionAC.npz", "HCRL-Primary.npz"],
                "HCRL_Backup": ["HCRL_Backup.npz", "HCRL_Backup_OptionAC.npz", "HCRL-Backup.npz"],
            }
            p = Path(path).expanduser()
            if p.is_file():
                print(f"[Weight load warning] HCRL-Oracle expects a run directory. File path {p} will be loaded as HCRL_Primary only.")
                loaded["HCRL_Primary"] = _load_one_model(brain.get("HCRL_Primary"), p, "HCRL_Primary", args)
            else:
                for key, names in mapping.items():
                    candidates = _candidate_checkpoint_files(p, names)
                    if candidates:
                        loaded[key] = _load_one_model(brain.get(key), candidates[0], key, args)
                    else:
                        print(f"[Weight load warning] missing {key} checkpoint under {p}; looked for {names}")
            continue
        model_key = method
        candidates = _candidate_checkpoint_files(path, [f"{method}.npz", f"{method.replace('-', '_')}.npz"])
        if not candidates:
            print(f"[Weight load warning] checkpoint not found for {method}: {path}")
            continue
        loaded[method] = _load_one_model(brain.get(model_key), candidates[0], method, args)
    if getattr(args, "Eval_Only", False):
        for model in brain.values():
            if hasattr(model, "set_epsilon"):
                # This affects stochastic choose_action, but eval code also uses choose_best_action.
                model.set_epsilon(1.0)
    return loaded


def _choose_value_policy(model, state, mask, args):
    return model.choose_best_action(state, mask) if getattr(args, "Greedy_Eval", False) and hasattr(model, "choose_best_action") else model.choose_action(state, mask)


def maybe_warm_start(brain, args, episode, flags):
    if getattr(args, "Eval_Only", False):
        return
    if "COBRA-Oracle" in args.Baselines and not flags.get("cobra", False) and not args.COBRA_No_Teacher and episode >= args.COBRA_WarmStart_Episode:
        teacher = brain.get(args.COBRA_Teacher_Source)
        if teacher is not None and brain["COBRA-Oracle"].copy_from(teacher, copy_optimizer_state=False):
            brain["COBRA-Oracle"].set_epsilon(min(getattr(teacher, "epsilon", 0.2), 0.35))
            flags["cobra"] = True
            print(f"[COBRA] warm-started from {args.COBRA_Teacher_Source} at episode {episode}")
    if "HCRL-Oracle" in args.Baselines and not flags.get("hcrl", False) and not args.HCRL_No_Teacher and episode >= args.HCRL_WarmStart_Episode:
        teacher = brain.get(args.HCRL_Teacher_Source)
        if teacher is not None and brain["HCRL_Primary"].copy_from(teacher, copy_optimizer_state=False):
            brain["HCRL_Primary"].set_epsilon(min(getattr(teacher, "epsilon", 0.2), 0.30))
            flags["hcrl"] = True
            print(f"[HCRL] warm-started primary selector from {args.HCRL_Teacher_Source} at episode {episode}")


def summarize_episode(env, args, startP, episode=None):
    total_rewards = env.get_totalRewards(args.Baseline_num, startP)
    avg_resp = env.get_total_responseTs(args.Baseline_num, startP)
    succ = env.get_totalSuccess(args.Baseline_num, startP)
    succ_time = env.get_totalSuccessInTime(args.Baseline_num, startP)
    finish = env.get_totalTimes(args.Baseline_num, startP)
    cost = env.get_totalCost(args.Baseline_num, startP)
    cps = env.get_totalCostPerSuccess(args.Baseline_num, startP)
    mal = env.get_totalAssignedMaliciousRate(args.Baseline_num, startP)
    trusted = env.get_totalAssignedTrustedRate(args.Baseline_num, startP)
    audit_rate = env.get_totalAuditRate(args.Baseline_num, startP)
    audit_fail = env.get_totalAuditFailRate(args.Baseline_num, startP)
    audit_truth = env.get_totalAuditTruthMean(args.Baseline_num, startP)
    print(f"total performance (after {startP} requests):")
    for i, name in enumerate(args.Baselines):
        print(
            f"[{name}] reward: {total_rewards[i]:.2f} "
            f"avg_responseT: {avg_resp[i]:.3f} "
            f"success_rate: {succ[i] * 100:.2f}% "
            f"success_time_rate: {succ_time[i] * 100:.2f}% "
            f"finishT: {finish[i]:.2f} "
            f"Cost: {cost[i]:.3f} "
            f"cost_per_success: {cps[i]:.3f} "
            f"malicious_rate: {mal[i] * 100:.2f}% "
            f"trusted_rate: {trusted[i] * 100:.2f}% "
            f"audit_rate: {audit_rate[i] * 100:.2f}% "
            f"audit_fail_rate: {audit_fail[i] * 100:.2f}% "
            f"audit_truth_mean: {audit_truth[i]:.3f}"
        )
    for diag_name in ["PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle"]:
        if diag_name in args.Baselines:
            idx = args.Baselines.index(diag_name)
            print(
                f"[{diag_name} diagnostics] "
                f"primary_success_rate: {env.get_totalPrimarySuccessRate(args.Baseline_num, startP)[idx] * 100:.2f}% "
                f"backup_used_rate: {env.get_totalBackupUsedRate(args.Baseline_num, startP)[idx] * 100:.2f}% "
                f"backup_recovery_rate: {env.get_totalBackupRecoveryRate(args.Baseline_num, startP)[idx] * 100:.2f}% "
                f"conditional_backup_recovery_rate: {env.get_totalConditionalBackupRecoveryRate(args.Baseline_num, startP)[idx] * 100:.2f}% "
                f"backup_skipped_rate: {env.get_totalBackupSkippedRate(args.Baseline_num, startP)[idx] * 100:.2f}% "
                f"backup_score_mean: {env.get_totalBackupScoreMean(args.Baseline_num, startP)[idx]:.3f} "
                f"single_mode_rate: {env.get_totalHCRLSingleModeRate(args.Baseline_num, startP)[idx] * 100:.2f}% "
                f"serial_mode_rate: {env.get_totalHCRLSerialModeRate(args.Baseline_num, startP)[idx] * 100:.2f}% "
                f"parallel_mode_rate: {env.get_totalHCRLParallelModeRate(args.Baseline_num, startP)[idx] * 100:.2f}%"
            )


def collect_final_results(env, args, startP):
    rows = []
    arrays = {
        "reward": env.get_totalRewards(args.Baseline_num, startP),
        "avg_responseT": env.get_total_responseTs(args.Baseline_num, startP),
        "success_rate": env.get_totalSuccess(args.Baseline_num, startP),
        "success_time_rate": env.get_totalSuccessInTime(args.Baseline_num, startP),
        "finishT": env.get_totalTimes(args.Baseline_num, startP),
        "Cost": env.get_totalCost(args.Baseline_num, startP),
        "cost_per_success": env.get_totalCostPerSuccess(args.Baseline_num, startP),
        "malicious_rate": env.get_totalAssignedMaliciousRate(args.Baseline_num, startP),
        "normal_rate": env.get_totalAssignedNormalRate(args.Baseline_num, startP),
        "trusted_rate": env.get_totalAssignedTrustedRate(args.Baseline_num, startP),
        "primary_success_rate": env.get_totalPrimarySuccessRate(args.Baseline_num, startP),
        "backup_used_rate": env.get_totalBackupUsedRate(args.Baseline_num, startP),
        "backup_recovery_rate": env.get_totalBackupRecoveryRate(args.Baseline_num, startP),
        "conditional_backup_recovery_rate": env.get_totalConditionalBackupRecoveryRate(args.Baseline_num, startP),
        "backup_skipped_rate": env.get_totalBackupSkippedRate(args.Baseline_num, startP),
        "backup_score_mean": env.get_totalBackupScoreMean(args.Baseline_num, startP),
        "single_mode_rate": env.get_totalHCRLSingleModeRate(args.Baseline_num, startP),
        "serial_mode_rate": env.get_totalHCRLSerialModeRate(args.Baseline_num, startP),
        "parallel_mode_rate": env.get_totalHCRLParallelModeRate(args.Baseline_num, startP),
        "audit_rate": env.get_totalAuditRate(args.Baseline_num, startP),
        "audit_pass_rate": env.get_totalAuditPassRate(args.Baseline_num, startP),
        "audit_fail_rate": env.get_totalAuditFailRate(args.Baseline_num, startP),
        "audit_truth_mean": env.get_totalAuditTruthMean(args.Baseline_num, startP),
    }
    for i, name in enumerate(args.Baselines):
        row = {"method": name}
        for k, v in arrays.items():
            row[k] = float(v[i])
        rows.append(row)
    return rows


# Main experiment.
env = SchedulingEnv(args)
brain = build_models(env, args)
loaded_weights = load_requested_weights(brain, args)
if loaded_weights:
    print(f"[Weight load summary] {loaded_weights}")
flags = {"cobra": False, "hcrl": False}
eval_start = min(2000, max(0, args.Request_Num // 2))
last_results = None

for episode in range(args.Epoch):
    print(f"----------------------------Episode {episode} ----------------------------")
    env.reset(args)
    env.reset_reputation_factors()
    env.initial_reputation()
    maybe_warm_start(brain, args, episode, flags)

    # Previous transition buffers.
    last = {}
    request_c = 1
    time_period = 1
    blor_c = 1
    performance_c = 0
    global_step = 0
    while True:
        if request_c % args.Time_Period_Size == 0:
            time_period += 1
            for policy_name in args.Baselines:
                env.update_reputation(env.get_reputation_factors(policy_name), time_period, policy_name)
            env.reset_reputation_factors()

        global_step += 1
        finish, request_attrs = env.workload(request_c)
        mask = env.get_action_mask(request_attrs)

        if "Random" in args.Baselines:
            a = brain["others"].random_choose_action(mask)
            env.feedback(request_attrs, a, "Random")
        if "Round-Robin" in args.Baselines:
            a = brain["others"].RR_choose_action(request_c, mask)
            env.feedback(request_attrs, a, "Round-Robin")
        if "Earliest" in args.Baselines:
            a = brain["others"].early_choose_action(env.get_oracle_idleT("Earliest"), mask)
            env.feedback(request_attrs, a, "Earliest")

        if "DQN" in brain:
            s = env.getState(request_attrs, "DQN")
            if not args.Eval_Only and "DQN" in last:
                brain["DQN"].store_transition(last["DQN"][0], last["DQN"][1], last["DQN"][2], s, mask)
            a = _choose_value_policy(brain["DQN"], s, mask, args)
            r = env.feedback(request_attrs, a, "DQN")
            if not args.Eval_Only and global_step > args.Dqn_start_learn and global_step % args.Dqn_learn_interval == 0:
                brain["DQN"].learn()
            if not args.Eval_Only:
                last["DQN"] = (s, a, r, mask)

        if "BLOR" in args.Baselines:
            start_counter = (blor_c - 1) * 200
            rr_counter = start_counter + min(15, env.actionNum)
            end_counter = blor_c * 200
            if start_counter < request_c <= rr_counter:
                a = brain["others"].RR_choose_action(request_c, mask)
            elif rr_counter < request_c <= end_counter:
                req = env.get_request_num("BLOR")
                suc = env.get_successful_validation("BLOR")
                fail = np.maximum(req - suc, 0)
                b = BLOR(suc, fail, env.oracleCost, seed=args.Seed + request_c)
                score = b.get_oracles(suc, fail, env.oracleCost)
                score[~mask] = -1e9
                a = b.choose_action(score)
            else:
                a = brain["others"].RR_choose_action(request_c, mask)
            env.feedback(request_attrs, a, "BLOR")
            if request_c % 200 == 0:
                env.reset_reputation_factors_BLOR()
                blor_c += 1

        if "SemiGreedy" in args.Baselines:
            rewards, cost = env.feedback_PSG_FWA(request_attrs, "SemiGreedy")
            a = brain["others"].PSG_choose_action(rewards, cost, mask)
            env.feedback(request_attrs, a, "SemiGreedy")

        if "PPO" in brain:
            s = env.getState(request_attrs, "PPO")
            if args.Greedy_Eval:
                a = brain["PPO"].choose_best_action(s, mask); prob = 1.0
            else:
                a, prob = brain["PPO"].choose_action(s, mask)
            r = env.feedback(request_attrs, a, "PPO")
            if not args.Eval_Only:
                brain["PPO"].store_transition(s, a, r, prob, mask)
                if global_step > args.PPO_start_learn and global_step % args.PPO_learn_interval == 0:
                    brain["PPO"].learn()

        if "RA-DDQN" in brain:
            s = env.getState(request_attrs, "RA-DDQN")
            if not args.Eval_Only and "RA-DDQN" in last:
                brain["RA-DDQN"].store_transition(last["RA-DDQN"][0], last["RA-DDQN"][1], last["RA-DDQN"][2], s, mask)
            a = _choose_value_policy(brain["RA-DDQN"], s, mask, args)
            r = env.feedback(request_attrs, a, "RA-DDQN")
            if not args.Eval_Only and global_step > args.RA_start_learn and global_step % args.RA_learn_interval == 0:
                brain["RA-DDQN"].learn()
            if not args.Eval_Only:
                last["RA-DDQN"] = (s, a, r, mask)

        if "PB-SafeDQN" in brain:
            s = env.getState(request_attrs, "PB-SafeDQN")
            if not args.Eval_Only and "PB-SafeDQN" in last:
                brain["PB-SafeDQN"].store_transition(last["PB-SafeDQN"][0], last["PB-SafeDQN"][1], last["PB-SafeDQN"][2], s, mask)
            a = _choose_value_policy(brain["PB-SafeDQN"], s, mask, args)
            r = env.feedback_primary_backup(request_attrs, a, "PB-SafeDQN")
            if not args.Eval_Only and global_step > args.PB_start_learn and global_step % args.PB_learn_interval == 0:
                brain["PB-SafeDQN"].learn()
            if not args.Eval_Only:
                last["PB-SafeDQN"] = (s, a, r, mask)

        if "COBRA-Oracle" in brain:
            s = env.getState(request_attrs, "COBRA-Oracle")
            if not args.Eval_Only and "COBRA-Oracle" in last:
                brain["COBRA-Oracle"].store_transition(last["COBRA-Oracle"][0], last["COBRA-Oracle"][1], last["COBRA-Oracle"][2], s, mask)
            teacher_action = None
            if not args.Eval_Only and not args.COBRA_No_Teacher and episode < args.COBRA_Teacher_Guidance_Episodes:
                teacher = brain.get(args.COBRA_Teacher_Source)
                if teacher is not None:
                    teacher_action = teacher.choose_best_action(s, mask)
            teacher_prob = 0.0
            if teacher_action is not None:
                frac = max(0.0, 1.0 - episode / max(args.COBRA_Teacher_Guidance_Episodes, 1))
                teacher_prob = max(args.COBRA_Min_Teacher_Prob, args.COBRA_Teacher_Start_Prob * frac)
            a = int(teacher_action) if teacher_action is not None and np.random.rand() < teacher_prob else _choose_value_policy(brain["COBRA-Oracle"], s, mask, args)
            r = env.feedback_primary_backup(request_attrs, a, "COBRA-Oracle")
            if not args.Eval_Only and global_step > args.COBRA_start_learn and global_step % args.COBRA_learn_interval == 0:
                brain["COBRA-Oracle"].learn()
            if not args.Eval_Only:
                last["COBRA-Oracle"] = (s, a, r, mask)

        if "HCRL-Oracle" in args.Baselines:
            s_primary = env.getState(request_attrs, "HCRL-Oracle")
            primary_mask = env.get_action_mask(request_attrs)
            # Primary selection, teacher-guided early.
            teacher_action = None
            if not args.Eval_Only and not args.HCRL_No_Teacher and episode < args.HCRL_Teacher_Guidance_Episodes:
                teacher = brain.get(args.HCRL_Teacher_Source)
                if teacher is not None:
                    teacher_action = teacher.choose_best_action(s_primary, primary_mask)
            teacher_prob = 0.0
            if teacher_action is not None:
                frac = max(0.0, 1.0 - episode / max(args.HCRL_Teacher_Guidance_Episodes, 1))
                teacher_prob = max(args.HCRL_Min_Teacher_Prob, args.HCRL_Teacher_Start_Prob * frac)
            primary_action = int(teacher_action) if teacher_action is not None and np.random.rand() < teacher_prob else _choose_value_policy(brain["HCRL_Primary"], s_primary, primary_mask, args)

            s_mode = env.get_hcrl_mode_state(request_attrs, "HCRL-Oracle")
            mode_mask = env.get_hcrl_mode_mask(request_attrs, "HCRL-Oracle", primary_action)
            if getattr(args, "HCRL_Fixed_Single_Mode", False):
                mode_action = 0
            elif getattr(args, "HCRL_Fixed_Parallel_Mode", False):
                mode_action = args.HCRL_Mode_Names.index("parallel_safe") if "parallel_safe" in args.HCRL_Mode_Names else len(args.HCRL_Mode_Names) - 1
            else:
                mode_action = _choose_value_policy(brain["HCRL_Mode"], s_mode, mode_mask, args)

            backup_mask = env.get_backup_action_mask(request_attrs, primary_action)
            mode_name = args.HCRL_Mode_Names[int(mode_action)]
            if mode_name in ["single_cost", "single_safe"]:
                backup_action = -1
            elif getattr(args, "HCRL_Random_Backup", False):
                valid = np.where(backup_mask)[0]
                backup_action = int(np.random.choice(valid)) if valid.size else -1
            else:
                backup_teacher_prob = 0.0
                if episode < args.HCRL_Backup_Guidance_Episodes:
                    frac_b = max(0.0, 1.0 - episode / max(args.HCRL_Backup_Guidance_Episodes, 1))
                    backup_teacher_prob = max(args.HCRL_Backup_Min_Prob, args.HCRL_Backup_Start_Prob * frac_b)
                if (not args.Eval_Only) and np.random.rand() < backup_teacher_prob:
                    backup_action = env.choose_backup_oracle(request_attrs, primary_action, "HCRL-Oracle")
                else:
                    backup_action = _choose_value_policy(brain["HCRL_Backup"], s_primary, backup_mask, args)

            # Store previous transitions with current states.
            if not args.Eval_Only and "HCRL_Mode" in last:
                brain["HCRL_Mode"].store_transition(last["HCRL_Mode"][0], last["HCRL_Mode"][1], last["HCRL_Mode"][2], s_mode, mode_mask, last["HCRL_Mode"][3])
            if not args.Eval_Only and "HCRL_Primary" in last:
                brain["HCRL_Primary"].store_transition(last["HCRL_Primary"][0], last["HCRL_Primary"][1], last["HCRL_Primary"][2], s_primary, primary_mask, last["HCRL_Primary"][3])
            if not args.Eval_Only and "HCRL_Backup" in last:
                brain["HCRL_Backup"].store_transition(last["HCRL_Backup"][0], last["HCRL_Backup"][1], last["HCRL_Backup"][2], s_primary, backup_mask, last["HCRL_Backup"][3])

            feedback = env.feedback_hcrl(request_attrs, mode_action, primary_action, backup_action, "HCRL-Oracle")
            if not args.Eval_Only and global_step > args.HCRL_start_learn and global_step % args.HCRL_Mode_learn_interval == 0:
                brain["HCRL_Mode"].learn()
            if not args.Eval_Only and global_step > args.HCRL_start_learn and global_step % args.HCRL_learn_interval == 0:
                brain["HCRL_Primary"].learn()
            if not args.Eval_Only and global_step > args.HCRL_start_learn and global_step % args.HCRL_Backup_learn_interval == 0:
                brain["HCRL_Backup"].learn()
            if not args.Eval_Only:
                last["HCRL_Mode"] = (s_mode, mode_action, feedback["mode_reward"], mode_mask)
                last["HCRL_Primary"] = (s_primary, primary_action, feedback["primary_reward"], primary_mask)
                if mode_name not in ["single_cost", "single_safe"] and backup_action >= 0:
                    last["HCRL_Backup"] = (s_primary, backup_action, feedback["backup_reward"], backup_mask)

        if request_c % 500 == 0:
            env.get_accumulateRewards(args.Baseline_num, performance_c, request_c)
            performance_c = request_c
        request_c += 1
        if finish:
            break

    summarize_episode(env, args, eval_start, episode)
    last_results = collect_final_results(env, args, eval_start)

# Save final results.
if last_results is None:
    last_results = collect_final_results(env, args, eval_start)
pd.DataFrame(last_results).to_csv(RUN_CSV_PATH, index=False, encoding="utf-8-sig")
with open(RUN_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump({"run_id": RUN_ID, "args": vars(args), "results": last_results}, f, ensure_ascii=False, indent=2, default=str)
print(f"Saved final results CSV: {RUN_CSV_PATH}")
print(f"Saved final results JSON: {RUN_JSON_PATH}")

# Save lightweight checkpoints where available.
for key, model in brain.items():
    if hasattr(model, "save_model"):
        try:
            model.save_model(str(RUN_DIR / f"{key}.npz"), metadata={"run_id": RUN_ID, "method": key})
        except Exception as e:
            print(f"[Warning] failed to save {key}: {e}")
