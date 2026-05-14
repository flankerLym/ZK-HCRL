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

from dynamic_malicious_runtime import (
    extract_dynamic_malicious_args,
    attach_dynamic_malicious_args,
    install_dynamic_malicious_training,
)

# Strip dynamic-malicious args before the original param_parser.py sees sys.argv.
_dyn_args, _clean_argv = extract_dynamic_malicious_args(sys.argv)
sys.argv = _clean_argv

from env import SchedulingEnv
from model import baseline_DQN, baseline_PPO, DuelingDoubleDQN, OptionActorCritic, baselines, BLOR
from utils import get_args

# Install dynamic malicious monkey-patch before environment construction.
install_dynamic_malicious_training(SchedulingEnv)


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
args = attach_dynamic_malicious_args(args, _dyn_args)
sys.stdout, sys.stderr = _pre_stdout, _pre_stderr

np.random.seed(args.Seed)
random.seed(args.Seed)

now = datetime.now()
tag = str(getattr(args, "Run_Tag", "")).strip()
tag_part = f"_{tag}" if tag else ""
dyn_part = "_DynMal" if getattr(args, "Dynamic_Malicious_Training", False) else ""
RUN_ID = f"{now.year%100}_{now.month}_{now.day}_{now.hour:02d}_{now.minute:02d}_Epoch{args.Epoch}_Req{args.Request_Num}_{args.Scenario}_Seed{args.Seed}{dyn_part}{tag_part}"
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
if getattr(args, "Dynamic_Malicious_Training", False):
    print(
        "[Dynamic malicious training] enabled: "
        f"refresh={args.Dynamic_Malicious_Refresh}, "
        f"refresh_periods={args.Dynamic_Malicious_Refresh_Periods}, "
        f"ratio={args.Dynamic_Malicious_Ratio}, "
        f"count={args.Dynamic_Malicious_Count}, "
        f"strategy={args.Dynamic_Malicious_Strategy}, "
        f"profile_strength={args.Dynamic_Malicious_Profile_Strength}"
    )


def _set_if_absent_or_lower(current, target):
    """Compatibility helper kept for future CLI extension."""
    try:
        return float(current)
    except Exception:
        return float(target)


def apply_dynamic_hcrl_adaptation(args):
    """Success-oriented HCRL tuning for dynamic malicious-oracle training.

    v4 balanced still tended to over-penalize audit/risk, which pushed HCRL into
    low-quality backup selection and reduced final success rate. This v5 preset
    makes final success, validation-aware success, and effective backup recovery
    the dominant reward signals, while keeping a moderate malicious-oracle
    penalty so the policy still avoids active attackers.
    """
    if not bool(getattr(args, "Dynamic_Malicious_Training", False)):
        return args
    if bool(getattr(args, "Disable_Dynamic_HCRL_Tune", False)):
        print("[Dynamic HCRL tune] disabled")
        return args

    mode = str(getattr(args, "Dynamic_HCRL_Tune_Mode", "success")).lower()
    if mode not in {"success", "balanced", "safe", "aggressive"}:
        mode = "success"

    # DQN teacher was trained in a different/non-dynamic regime and can bias HCRL
    # toward fixed-oracle behavior. Keep it disabled for dynamic attackers.
    args.HCRL_No_Teacher = True
    args.HCRL_Teacher_Source = "none"
    args.HCRL_Teacher_Start_Prob = 0.0
    args.HCRL_Min_Teacher_Prob = 0.0
    args.HCRL_Teacher_Guidance_Episodes = 0

    # Slower malicious curriculum: v3 reached full attack by episode 5 and HCRL
    # collapsed. These defaults let it learn useful backup/reputation evidence.
    if not bool(getattr(args, "Disable_Dynamic_Malicious_Curriculum", False)):
        args.Dynamic_Malicious_Start_Ratio = float(getattr(args, "Dynamic_Malicious_Start_Ratio", 0.06))
        args.Dynamic_Malicious_Start_Strength = float(getattr(args, "Dynamic_Malicious_Start_Strength", 0.45))
        args.Dynamic_Malicious_Warmup_Episodes = max(int(getattr(args, "Dynamic_Malicious_Warmup_Episodes", 18)), 18)

    if mode == "success":
        cfg = dict(
            # Moderate risk: malicious avoidance remains meaningful, but it no
            # longer dominates final success and validation-aware success.
            risk_lambda=0.88, cost_lambda=0.55, risk_budget=0.085,
            primary_mal_pen=0.95, backup_mal_pen=1.05, est_risk_pen=0.34,
            # Do not punish backup merely for being used; punish low-value / bad
            # recovery indirectly through final success and recovery signals.
            backup_used_pen=0.025, unnecessary_pen=0.045, total_cost_pen=0.10,
            # Looser backup gate to prevent backup_score collapse. This should
            # increase effective recovery rather than forcing near-constant but
            # low-quality backup behavior.
            backup_min_score=0.055, backup_max_risk=0.62, backup_risk_margin=0.03,
            backup_cost_cap=1.22, recovery_cost_cap=1.45,
            # Main reward target: final success and useful backup recovery.
            final_success=1.08, success_gain=1.05, primary_success=0.62,
            backup_recovery=1.38, trusted_bonus=0.12, backup_trust=0.13,
            # Audit remains useful, but lower its weight to avoid suppressing the
            # whole oracle pool when attacks rotate dynamically.
            audit_weight=0.22, audit_risk_rate=0.075, audit_fail_pen=0.045,
            audit_reward_pen=0.16, low_truth=0.36, high_risk=0.70, cooldown=0.045,
            entropy=0.075,
            # Keep backup exploration longer so the backup branch can learn which
            # alternatives truly recover failed primaries.
            backup_guidance_eps=18, backup_start_prob=0.72, backup_min_prob=0.08,
        )
    elif mode == "aggressive":
        cfg = dict(
            risk_lambda=1.35, cost_lambda=0.90, risk_budget=0.055,
            primary_mal_pen=1.35, backup_mal_pen=1.50, est_risk_pen=0.62,
            backup_used_pen=0.10, unnecessary_pen=0.16, total_cost_pen=0.24,
            backup_min_score=0.16, backup_max_risk=0.42, backup_risk_margin=0.08,
            backup_cost_cap=1.02, recovery_cost_cap=1.20,
            final_success=0.66, success_gain=0.72, primary_success=0.42,
            backup_recovery=0.92, trusted_bonus=0.18, backup_trust=0.18,
            audit_weight=0.36, audit_risk_rate=0.13, audit_fail_pen=0.09,
            audit_reward_pen=0.34, low_truth=0.44, high_risk=0.60, cooldown=0.12,
            entropy=0.07,
            backup_guidance_eps=8, backup_start_prob=0.50, backup_min_prob=0.03,
        )
    elif mode == "safe":
        cfg = dict(
            risk_lambda=1.20, cost_lambda=0.78, risk_budget=0.065,
            primary_mal_pen=1.20, backup_mal_pen=1.35, est_risk_pen=0.52,
            backup_used_pen=0.06, unnecessary_pen=0.08, total_cost_pen=0.16,
            backup_min_score=0.10, backup_max_risk=0.52, backup_risk_margin=0.05,
            backup_cost_cap=1.10, recovery_cost_cap=1.30,
            final_success=0.74, success_gain=0.82, primary_success=0.46,
            backup_recovery=1.05, trusted_bonus=0.16, backup_trust=0.17,
            audit_weight=0.28, audit_risk_rate=0.11, audit_fail_pen=0.07,
            audit_reward_pen=0.24, low_truth=0.40, high_risk=0.64, cooldown=0.08,
            entropy=0.06,
            backup_guidance_eps=10, backup_start_prob=0.58, backup_min_prob=0.04,
        )
    else:  # balanced, recommended default
        cfg = dict(
            risk_lambda=1.15, cost_lambda=0.80, risk_budget=0.065,
            primary_mal_pen=1.15, backup_mal_pen=1.28, est_risk_pen=0.50,
            backup_used_pen=0.07, unnecessary_pen=0.10, total_cost_pen=0.18,
            backup_min_score=0.12, backup_max_risk=0.50, backup_risk_margin=0.06,
            backup_cost_cap=1.08, recovery_cost_cap=1.25,
            final_success=0.72, success_gain=0.78, primary_success=0.44,
            backup_recovery=0.98, trusted_bonus=0.16, backup_trust=0.16,
            audit_weight=0.30, audit_risk_rate=0.12, audit_fail_pen=0.08,
            audit_reward_pen=0.28, low_truth=0.42, high_risk=0.62, cooldown=0.10,
            entropy=0.06,
            backup_guidance_eps=8, backup_start_prob=0.50, backup_min_prob=0.03,
        )

    args.HCRL_Backup_Guidance_Episodes = int(cfg["backup_guidance_eps"])
    args.HCRL_Backup_Start_Prob = float(cfg["backup_start_prob"])
    args.HCRL_Backup_Min_Prob = float(cfg["backup_min_prob"])

    args.HCRL_Primary_Malicious_Penalty = float(cfg["primary_mal_pen"])
    args.HCRL_Backup_Malicious_Penalty = float(cfg["backup_mal_pen"])
    args.HCRL_Estimated_Risk_Penalty = float(cfg["est_risk_pen"])
    args.HCRL_Lambda_Risk = float(cfg["risk_lambda"])
    args.HCRL_Risk_Budget = float(cfg["risk_budget"])

    args.HCRL_Backup_Used_Penalty = float(cfg["backup_used_pen"])
    args.HCRL_Unnecessary_Backup_Penalty = float(cfg["unnecessary_pen"])
    args.HCRL_Total_Cost_Penalty = float(cfg["total_cost_pen"])
    args.HCRL_Lambda_Cost = float(cfg["cost_lambda"])
    args.HCRL_Safety_Min_Backup_Score = float(cfg["backup_min_score"])
    args.HCRL_Backup_Max_Estimated_Risk = float(cfg["backup_max_risk"])
    args.HCRL_Backup_Risk_Margin = float(cfg["backup_risk_margin"])
    args.HCRL_Backup_Cost_Cap = float(cfg["backup_cost_cap"])
    args.HCRL_Recovery_Cost_Hard_Cap = float(cfg["recovery_cost_cap"])

    args.HCRL_Final_Success_Bonus = float(cfg["final_success"])
    args.HCRL_Success_Gain_Bonus = float(cfg["success_gain"])
    args.HCRL_Primary_Success_Bonus = float(cfg["primary_success"])
    args.HCRL_Backup_Recovery_Bonus = float(cfg["backup_recovery"])
    args.HCRL_Trusted_Selection_Bonus = float(cfg["trusted_bonus"])
    args.HCRL_Backup_Trust_Bonus = float(cfg["backup_trust"])

    args.Audit_Weight_In_Reputation = float(cfg["audit_weight"])
    args.Audit_Risk_Rate = float(cfg["audit_risk_rate"])
    args.Audit_Fail_Penalty = float(cfg["audit_fail_pen"])
    args.Audit_Risk_Reward_Penalty = float(cfg["audit_reward_pen"])
    args.Audit_Low_Truth_Threshold = float(cfg["low_truth"])
    args.Audit_High_Risk_Threshold = float(cfg["high_risk"])
    args.Audit_Cooldown_Penalty = float(cfg["cooldown"])

    args.HCRL_AC_Entropy = max(float(getattr(args, "HCRL_AC_Entropy", 0.05)), float(cfg["entropy"]))
    args.Dqn_memory_size = max(int(getattr(args, "Dqn_memory_size", 3000)), 6000)

    print(f"[Dynamic HCRL tune] enabled mode={mode}: success-oriented reward, no DQN teacher")
    print(
        f"[Dynamic HCRL tune] risk_lambda={args.HCRL_Lambda_Risk}, cost_lambda={args.HCRL_Lambda_Cost}, "
        f"success_bonus={args.HCRL_Final_Success_Bonus}, recovery_bonus={args.HCRL_Backup_Recovery_Bonus}, "
        f"backup_used_penalty={args.HCRL_Backup_Used_Penalty}, backup_min_score={args.HCRL_Safety_Min_Backup_Score}, "
        f"audit_weight={args.Audit_Weight_In_Reputation}, warmup={getattr(args, 'Dynamic_Malicious_Warmup_Episodes', 'NA')}"
    )
    return args


args = apply_dynamic_hcrl_adaptation(args)


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


def maybe_warm_start(brain, args, episode, flags):
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
    if getattr(args, "Dynamic_Malicious_Training", False):
        active = getattr(env, "active_malicious_oracles", [])
        print(f"[Dynamic malicious diagnostics] generation={getattr(env, 'dynamic_malicious_generation', -1)} active={active}")


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
args.Dynamic_Malicious_Current_Episode = 0
env = SchedulingEnv(args)
brain = build_models(env, args)
flags = {"cobra": False, "hcrl": False}
eval_start = min(2000, max(0, args.Request_Num // 2))
last_results = None
last_dynamic_history = []

for episode in range(args.Epoch):
    print(f"----------------------------Episode {episode} ----------------------------")
    args.Dynamic_Malicious_Current_Episode = int(episode)
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
            if hasattr(env, "maybe_refresh_dynamic_malicious"):
                env.maybe_refresh_dynamic_malicious(trigger="period", time_period=time_period, request_id=request_c)

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
            if "DQN" in last:
                brain["DQN"].store_transition(last["DQN"][0], last["DQN"][1], last["DQN"][2], s, mask)
            a = brain["DQN"].choose_action(s, mask)
            r = env.feedback(request_attrs, a, "DQN")
            if global_step > args.Dqn_start_learn and global_step % args.Dqn_learn_interval == 0:
                brain["DQN"].learn()
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
            a, prob = brain["PPO"].choose_action(s, mask)
            r = env.feedback(request_attrs, a, "PPO")
            brain["PPO"].store_transition(s, a, r, prob, mask)
            if global_step > args.PPO_start_learn and global_step % args.PPO_learn_interval == 0:
                brain["PPO"].learn()

        if "RA-DDQN" in brain:
            s = env.getState(request_attrs, "RA-DDQN")
            if "RA-DDQN" in last:
                brain["RA-DDQN"].store_transition(last["RA-DDQN"][0], last["RA-DDQN"][1], last["RA-DDQN"][2], s, mask)
            a = brain["RA-DDQN"].choose_action(s, mask)
            r = env.feedback(request_attrs, a, "RA-DDQN")
            if global_step > args.RA_start_learn and global_step % args.RA_learn_interval == 0:
                brain["RA-DDQN"].learn()
            last["RA-DDQN"] = (s, a, r, mask)

        if "PB-SafeDQN" in brain:
            s = env.getState(request_attrs, "PB-SafeDQN")
            if "PB-SafeDQN" in last:
                brain["PB-SafeDQN"].store_transition(last["PB-SafeDQN"][0], last["PB-SafeDQN"][1], last["PB-SafeDQN"][2], s, mask)
            a = brain["PB-SafeDQN"].choose_action(s, mask)
            r = env.feedback_primary_backup(request_attrs, a, "PB-SafeDQN")
            if global_step > args.PB_start_learn and global_step % args.PB_learn_interval == 0:
                brain["PB-SafeDQN"].learn()
            last["PB-SafeDQN"] = (s, a, r, mask)

        if "COBRA-Oracle" in brain:
            s = env.getState(request_attrs, "COBRA-Oracle")
            if "COBRA-Oracle" in last:
                brain["COBRA-Oracle"].store_transition(last["COBRA-Oracle"][0], last["COBRA-Oracle"][1], last["COBRA-Oracle"][2], s, mask)
            teacher_action = None
            if not args.COBRA_No_Teacher and episode < args.COBRA_Teacher_Guidance_Episodes:
                teacher = brain.get(args.COBRA_Teacher_Source)
                if teacher is not None:
                    teacher_action = teacher.choose_best_action(s, mask)
            teacher_prob = 0.0
            if teacher_action is not None:
                frac = max(0.0, 1.0 - episode / max(args.COBRA_Teacher_Guidance_Episodes, 1))
                teacher_prob = max(args.COBRA_Min_Teacher_Prob, args.COBRA_Teacher_Start_Prob * frac)
            a = int(teacher_action) if teacher_action is not None and np.random.rand() < teacher_prob else brain["COBRA-Oracle"].choose_action(s, mask)
            r = env.feedback_primary_backup(request_attrs, a, "COBRA-Oracle")
            if global_step > args.COBRA_start_learn and global_step % args.COBRA_learn_interval == 0:
                brain["COBRA-Oracle"].learn()
            last["COBRA-Oracle"] = (s, a, r, mask)

        if "HCRL-Oracle" in args.Baselines:
            s_primary = env.getState(request_attrs, "HCRL-Oracle")
            primary_mask = env.get_action_mask(request_attrs)

            teacher_action = None
            if not args.HCRL_No_Teacher and episode < args.HCRL_Teacher_Guidance_Episodes:
                teacher = brain.get(args.HCRL_Teacher_Source)
                if teacher is not None:
                    teacher_action = teacher.choose_best_action(s_primary, primary_mask)
            teacher_prob = 0.0
            if teacher_action is not None:
                frac = max(0.0, 1.0 - episode / max(args.HCRL_Teacher_Guidance_Episodes, 1))
                teacher_prob = max(args.HCRL_Min_Teacher_Prob, args.HCRL_Teacher_Start_Prob * frac)
            primary_action = int(teacher_action) if teacher_action is not None and np.random.rand() < teacher_prob else brain["HCRL_Primary"].choose_action(s_primary, primary_mask)

            s_mode = env.get_hcrl_mode_state(request_attrs, "HCRL-Oracle")
            mode_mask = env.get_hcrl_mode_mask(request_attrs, "HCRL-Oracle", primary_action)
            if getattr(args, "HCRL_Fixed_Single_Mode", False):
                mode_action = 0
            elif getattr(args, "HCRL_Fixed_Parallel_Mode", False):
                mode_action = args.HCRL_Mode_Names.index("parallel_safe") if "parallel_safe" in args.HCRL_Mode_Names else len(args.HCRL_Mode_Names) - 1
            else:
                mode_action = brain["HCRL_Mode"].choose_action(s_mode, mode_mask)

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
                if np.random.rand() < backup_teacher_prob:
                    backup_action = env.choose_backup_oracle(request_attrs, primary_action, "HCRL-Oracle")
                else:
                    backup_action = brain["HCRL_Backup"].choose_action(s_primary, backup_mask)

            if "HCRL_Mode" in last:
                brain["HCRL_Mode"].store_transition(last["HCRL_Mode"][0], last["HCRL_Mode"][1], last["HCRL_Mode"][2], s_mode, mode_mask, last["HCRL_Mode"][3])
            if "HCRL_Primary" in last:
                brain["HCRL_Primary"].store_transition(last["HCRL_Primary"][0], last["HCRL_Primary"][1], last["HCRL_Primary"][2], s_primary, primary_mask, last["HCRL_Primary"][3])
            if "HCRL_Backup" in last:
                brain["HCRL_Backup"].store_transition(last["HCRL_Backup"][0], last["HCRL_Backup"][1], last["HCRL_Backup"][2], s_primary, backup_mask, last["HCRL_Backup"][3])

            feedback = env.feedback_hcrl(request_attrs, mode_action, primary_action, backup_action, "HCRL-Oracle")
            if global_step > args.HCRL_start_learn and global_step % args.HCRL_Mode_learn_interval == 0:
                brain["HCRL_Mode"].learn()
            if global_step > args.HCRL_start_learn and global_step % args.HCRL_learn_interval == 0:
                brain["HCRL_Primary"].learn()
            if global_step > args.HCRL_start_learn and global_step % args.HCRL_Backup_learn_interval == 0:
                brain["HCRL_Backup"].learn()

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
    if getattr(args, "Dynamic_Malicious_Training", False):
        last_dynamic_history = list(getattr(env, "dynamic_malicious_history", []))

# Save final results.
if last_results is None:
    last_results = collect_final_results(env, args, eval_start)

pd.DataFrame(last_results).to_csv(RUN_CSV_PATH, index=False, encoding="utf-8-sig")
json_payload = {
    "run_id": RUN_ID,
    "args": vars(args),
    "results": last_results,
}
if getattr(args, "Dynamic_Malicious_Training", False):
    json_payload["dynamic_malicious_history_last_episode"] = last_dynamic_history
    json_payload["dynamic_malicious_final_active"] = list(getattr(env, "active_malicious_oracles", []))
with open(RUN_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(json_payload, f, ensure_ascii=False, indent=2, default=str)

print(f"Saved final results CSV: {RUN_CSV_PATH}")
print(f"Saved final results JSON: {RUN_JSON_PATH}")

# Save lightweight checkpoints where available.
for key, model in brain.items():
    if hasattr(model, "save_model"):
        try:
            model.save_model(str(RUN_DIR / f"{key}.npz"), metadata={"run_id": RUN_ID, "method": key})
        except Exception as e:
            print(f"[Warning] failed to save {key}: {e}")
