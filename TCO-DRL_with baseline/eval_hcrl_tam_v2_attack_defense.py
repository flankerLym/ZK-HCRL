# -*- coding: utf-8 -*-
"""
HCRL-TAM-v2 attack-defense evaluation for TCO-DRL baseline.

This script loads trained HCRL .npz checkpoints and evaluates ME / OOA / OSA
attack defense without retraining. It compares:
  - HCRL_no_mask: original service-type mask only
  - HCRL_TAM_v1: simple trust-threshold mask with loose fallback
  - HCRL_TAM_v2: robust trust mask with quarantine, hysteresis, dynamic trust decay,
                 strict fallback, and backup anti-risk filtering

Place this file in:
  TCO-DRL_with baseline/eval_hcrl_tam_v2_attack_defense.py

Run from TCO-DRL_with baseline, for example:
  python eval_hcrl_tam_v2_attack_defense.py --Weight_Dir "..\TCO-DRL_on blockchain\weight\audit_hcrl" --Attacks ME OOA OSA --Methods HCRL-Oracle --Scenario rl_harder --Seed 6 --Oracle_Num 30 --Oracles_Per_Type 10 --Service_Type_Num 3 --Request_Num 6000 --Max_Requests 3000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import get_args
from env import SchedulingEnv
from model import OptionActorCritic


# -----------------------------
# CLI helpers
# -----------------------------

def _parse_int_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        toks = []
        for x in value:
            toks.extend(str(x).replace(";", ",").split(","))
    else:
        toks = str(value).replace(";", ",").split(",")
    out = []
    for t in toks:
        t = str(t).strip()
        if t:
            out.append(int(t))
    return out


def _custom_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--Weight_Dir", type=str, default="../TCO-DRL_on blockchain/weight/audit_hcrl")
    p.add_argument("--Mode_Weight", type=str, default="HCRL_Mode.npz")
    p.add_argument("--Primary_Weight", type=str, default="HCRL_Primary.npz")
    p.add_argument("--Backup_Weight", type=str, default="HCRL_Backup.npz")
    p.add_argument("--Output_Dir_Attack", type=str, default="attack_hcrl_tam_v2_output")
    p.add_argument("--Attacks", nargs="+", default=["ME", "OOA", "OSA"], choices=["ME", "OOA", "OSA"])
    p.add_argument("--Variants", nargs="+", default=["HCRL_no_mask", "HCRL_TAM_v1", "HCRL_TAM_v2"],
                   choices=["HCRL_no_mask", "HCRL_TAM_v1", "HCRL_TAM_v2"])
    p.add_argument("--Attack_Oracles", type=str, default="4,19,29")
    p.add_argument("--Trusted_Reference_Oracles", type=str, default="3,14,24")
    p.add_argument("--Attack_Start_Period", type=int, default=3)
    p.add_argument("--Max_Requests", type=int, default=3000)
    p.add_argument("--Figure_Format", type=str, default="png", choices=["png", "jpg", "svg", "pdf"])

    # Attack behavior.
    p.add_argument("--ME_Validation_Prob", type=float, default=0.03)
    p.add_argument("--ME_Behavior_Probs", type=str, default="0.02,0.08,0.35,0.55")
    p.add_argument("--OOA_Off_Periods", type=int, default=1,
                   help="Malicious/off periods per cycle for OOA.")
    p.add_argument("--OOA_On_Periods", type=int, default=5,
                   help="Honest/on recovery periods per cycle for OOA.")
    p.add_argument("--OSA_Margin", type=float, default=0.08,
                   help="OSA attacks when trust exceeds threshold + margin; otherwise behaves cleanly.")

    # TAM-v1/v2 trust thresholds.
    p.add_argument("--Trust_Action_Threshold", type=float, default=0.45)
    p.add_argument("--Trust_Recover_Threshold", type=float, default=0.62)
    p.add_argument("--Trust_Dynamic_Threshold", type=float, default=0.42)
    p.add_argument("--Recover_Clean_Periods", type=int, default=5)
    p.add_argument("--Quarantine_Periods", type=int, default=8)
    p.add_argument("--Quarantine_Extend_Periods", type=int, default=5)
    p.add_argument("--Dynamic_Bad_Alpha", type=float, default=0.65,
                   help="Fast EWMA update after bad evidence.")
    p.add_argument("--Dynamic_Good_Alpha", type=float, default=0.06,
                   help="Slow EWMA recovery after good evidence.")
    p.add_argument("--Strict_Backup_Filter", action="store_true", default=True)
    p.add_argument("--Disable_Strict_Backup_Filter", action="store_true")
    p.add_argument("--Allow_Cross_Service_Fallback", action="store_true", default=True)
    p.add_argument("--Disable_Cross_Service_Fallback", action="store_true")
    p.add_argument("--TopK_Fallback", type=int, default=3)
    p.add_argument("--Safe_Mode_Override", action="store_true", default=True)
    p.add_argument("--Disable_Safe_Mode_Override", action="store_true")
    return p


def parse_all_args():
    custom, remaining = _custom_parser().parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()

    # Force HCRL-only if the user forgot method selection.
    if "HCRL-Oracle" not in getattr(args, "Baselines", []):
        args.Baselines = ["HCRL-Oracle"]
        args.Baseline_num = 1
        args.Use_HCRL = True
        args.Use_GNN_Encoder = True

    if custom.Disable_Strict_Backup_Filter:
        custom.Strict_Backup_Filter = False
    if custom.Disable_Cross_Service_Fallback:
        custom.Allow_Cross_Service_Fallback = False
    if custom.Disable_Safe_Mode_Override:
        custom.Safe_Mode_Override = False
    custom.Attack_Oracles = _parse_int_list(custom.Attack_Oracles)
    custom.Trusted_Reference_Oracles = _parse_int_list(custom.Trusted_Reference_Oracles)
    custom.ME_Behavior_Probs = np.asarray([float(x) for x in str(custom.ME_Behavior_Probs).split(",")], dtype=float)
    custom.ME_Behavior_Probs = custom.ME_Behavior_Probs / max(custom.ME_Behavior_Probs.sum(), 1e-8)
    return args, custom


# -----------------------------
# Model loading
# -----------------------------

def _resolve_weight(weight_dir: str, name: str) -> Path:
    p = Path(name)
    if not p.is_absolute():
        p = Path(weight_dir) / p
    return p.resolve()


def load_npz_checkpoint(model, path: Path, strict: bool = True):
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    data = np.load(str(path), allow_pickle=True)
    loaded = []
    for name in ["W1", "b1", "Wp", "bp", "Wv", "bv", "W2", "b2", "Wa", "ba"]:
        if name in data.files and hasattr(model, name):
            arr = np.asarray(data[name], dtype=np.float32)
            cur = getattr(model, name)
            if strict and hasattr(cur, "shape") and tuple(cur.shape) != tuple(arr.shape):
                raise ValueError(f"Shape mismatch for {path.name}:{name}: checkpoint {arr.shape} != model {cur.shape}")
            setattr(model, name, arr)
            loaded.append(name)
    if not loaded:
        raise ValueError(f"No compatible params in {path}. Keys={data.files}")


def build_hcrl_models(env: SchedulingEnv, args, custom):
    mode_dim = env.s_features + env.mode_extra_features
    n_modes = len(args.HCRL_Mode_Names)
    models = {
        "mode": OptionActorCritic(
            n_modes, mode_dim, hidden_units=max(64, args.Dqn_hidden // 2),
            scope="HCRL_Mode_OptionAC", learning_rate=args.HCRL_Mode_lr,
            memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
            entropy_coef=args.HCRL_AC_Entropy, value_coef=args.HCRL_AC_Value_Coef,
            reward_clip=args.Reward_Clip, seed=args.Seed + 3031,
        ),
        "primary": OptionActorCritic(
            env.actionNum, env.s_features, hidden_units=args.Dqn_hidden,
            scope="HCRL_Primary_OptionAC", learning_rate=args.HCRL_lr,
            memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
            entropy_coef=args.HCRL_AC_Entropy, value_coef=args.HCRL_AC_Value_Coef,
            reward_clip=args.Reward_Clip, seed=args.Seed + 4049,
        ),
        "backup": OptionActorCritic(
            env.actionNum, env.s_features, hidden_units=args.Dqn_hidden,
            scope="HCRL_Backup_OptionAC", learning_rate=args.HCRL_lr,
            memory_size=args.Dqn_memory_size, batch_size=args.Dqn_batch_size,
            entropy_coef=args.HCRL_AC_Entropy, value_coef=args.HCRL_AC_Value_Coef,
            reward_clip=args.Reward_Clip, seed=args.Seed + 5051,
        ),
    }
    load_npz_checkpoint(models["mode"], _resolve_weight(custom.Weight_Dir, custom.Mode_Weight), strict=True)
    load_npz_checkpoint(models["primary"], _resolve_weight(custom.Weight_Dir, custom.Primary_Weight), strict=True)
    load_npz_checkpoint(models["backup"], _resolve_weight(custom.Weight_Dir, custom.Backup_Weight), strict=True)
    return models


# -----------------------------
# Attack and defense states
# -----------------------------

@dataclass
class DefenseState:
    oracle_num: int
    threshold: float
    recover_threshold: float
    dynamic_threshold: float
    quarantine_periods: int
    recover_clean_periods: int
    dynamic_bad_alpha: float
    dynamic_good_alpha: float
    dynamic_trust: np.ndarray = field(init=False)
    quarantine_until: np.ndarray = field(init=False)
    clean_periods: np.ndarray = field(init=False)
    was_quarantined: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dynamic_trust = np.full(self.oracle_num, 0.50, dtype=float)
        self.quarantine_until = np.zeros(self.oracle_num, dtype=int)
        self.clean_periods = np.zeros(self.oracle_num, dtype=int)
        self.was_quarantined = np.zeros(self.oracle_num, dtype=bool)

    def trust_observation(self, env: SchedulingEnv, policy_name="HCRL-Oracle") -> np.ndarray:
        rep = env._effective_reputation_vector(policy_name)
        truth = env.audit_truth_score(policy_name)
        try:
            cooldown = env._audit_cooldown_fraction(policy_name)
        except Exception:
            cooldown = np.zeros(self.oracle_num, dtype=float)
        obs = 0.45 * rep + 0.35 * truth + 0.20 * (1.0 - cooldown)
        return np.clip(obs, 0.0, 1.0)

    def update_period(self, env: SchedulingEnv, period: int, policy_name="HCRL-Oracle"):
        obs = self.trust_observation(env, policy_name)
        better = obs >= self.dynamic_trust
        alpha = np.where(better, self.dynamic_good_alpha, self.dynamic_bad_alpha)
        self.dynamic_trust = np.clip((1.0 - alpha) * self.dynamic_trust + alpha * obs, 0.0, 1.0)

        clean = (obs >= self.recover_threshold) & (self.dynamic_trust >= max(self.dynamic_threshold, self.threshold))
        self.clean_periods[clean] += 1
        self.clean_periods[~clean] = 0

        bad = (obs < self.threshold) | (self.dynamic_trust < self.dynamic_threshold)
        for i in np.where(bad)[0]:
            self.quarantine_until[i] = max(self.quarantine_until[i], int(period) + self.quarantine_periods)
            self.was_quarantined[i] = True
            self.clean_periods[i] = 0

        # Hysteresis recovery: a previously quarantined oracle must cross a higher
        # threshold and keep a clean streak before re-entering the candidate set.
        for i in np.where(self.was_quarantined)[0]:
            if period < self.quarantine_until[i]:
                continue
            if self.clean_periods[i] >= self.recover_clean_periods:
                self.was_quarantined[i] = False
            else:
                self.quarantine_until[i] = max(self.quarantine_until[i], int(period) + 1)

    def allowed(self, env: SchedulingEnv, period: int, policy_name="HCRL-Oracle") -> np.ndarray:
        obs = self.trust_observation(env, policy_name)
        allowed = (obs >= self.threshold) & (self.dynamic_trust >= self.dynamic_threshold)
        quarantined = np.arange(self.oracle_num) < 0  # all False
        quarantined = np.asarray(period < self.quarantine_until, dtype=bool) | self.was_quarantined
        # Allow recovered nodes only after hysteresis.
        recovered = (obs >= self.recover_threshold) & (self.clean_periods >= self.recover_clean_periods) & (period >= self.quarantine_until)
        allowed = allowed & (~quarantined | recovered)
        return allowed.astype(bool)


def apply_attack_profile(env: SchedulingEnv, attack: str, period: int, custom, original_validation, original_behavior,
                         defense_state: DefenseState | None = None, policy_name="HCRL-Oracle"):
    env.oracleValidationProbs[:] = original_validation.copy()
    env.oracleBehaviorProbs[:, :] = original_behavior.copy()
    if period < custom.Attack_Start_Period:
        return

    attack = attack.upper()
    attack_oracles = [i for i in custom.Attack_Oracles if 0 <= i < env.oracleNum]
    malicious_now = []

    if attack == "ME":
        malicious_now = attack_oracles
    elif attack == "OOA":
        cycle = max(custom.OOA_Off_Periods + custom.OOA_On_Periods, 1)
        phase = (period - custom.Attack_Start_Period) % cycle
        malicious_now = attack_oracles if phase < custom.OOA_Off_Periods else []
    elif attack == "OSA":
        if defense_state is None:
            rep = env._effective_reputation_vector(policy_name)
        else:
            rep = defense_state.trust_observation(env, policy_name)
        malicious_now = [i for i in attack_oracles if rep[i] > custom.Trust_Action_Threshold + custom.OSA_Margin]

    for i in malicious_now:
        env.oracleValidationProbs[i] = custom.ME_Validation_Prob
        env.oracleBehaviorProbs[i] = custom.ME_Behavior_Probs.copy()


def current_period(request_id: int, time_period_size: int) -> int:
    return int(request_id // max(int(time_period_size), 1)) + 1


# -----------------------------
# Mask and decision logic
# -----------------------------

def _base_service_mask(env: SchedulingEnv, request_attrs) -> np.ndarray:
    return np.asarray(env.get_action_mask(request_attrs), dtype=bool)


def _combined_trust(env: SchedulingEnv, defense_state: DefenseState | None, policy_name="HCRL-Oracle") -> np.ndarray:
    rep = env._effective_reputation_vector(policy_name)
    truth = env.audit_truth_score(policy_name)
    try:
        cooldown = env._audit_cooldown_fraction(policy_name)
    except Exception:
        cooldown = np.zeros_like(rep)
    base = np.clip(0.45 * rep + 0.35 * truth + 0.20 * (1.0 - cooldown), 0.0, 1.0)
    if defense_state is not None:
        base = np.clip(0.60 * base + 0.40 * defense_state.dynamic_trust, 0.0, 1.0)
    return base


def trust_mask_v1(env: SchedulingEnv, request_attrs, custom, policy_name="HCRL-Oracle", exclude: Iterable[int] = ()):  # loose fallback
    base = _base_service_mask(env, request_attrs)
    rep = env._effective_reputation_vector(policy_name)
    mask = base & (rep >= custom.Trust_Action_Threshold)
    for x in exclude:
        if 0 <= int(x) < env.oracleNum:
            mask[int(x)] = False
    if not np.any(mask):
        mask = base.copy()
        for x in exclude:
            if 0 <= int(x) < env.oracleNum:
                mask[int(x)] = False
    if not np.any(mask):
        mask = np.ones(env.oracleNum, dtype=bool)
        for x in exclude:
            if 0 <= int(x) < env.oracleNum:
                mask[int(x)] = False
    return mask.astype(bool)


def trust_mask_v2(env: SchedulingEnv, request_attrs, custom, defense_state: DefenseState, period: int,
                  policy_name="HCRL-Oracle", exclude: Iterable[int] = ()) -> np.ndarray:
    service = _base_service_mask(env, request_attrs)
    allowed = defense_state.allowed(env, period, policy_name)
    trust = _combined_trust(env, defense_state, policy_name)

    exclude = [int(x) for x in exclude if 0 <= int(x) < env.oracleNum]
    mask = service & allowed
    for x in exclude:
        mask[x] = False
    if np.any(mask):
        return mask.astype(bool)

    # Strict fallback 1: allow high-trust cross-service oracle. This is deliberate:
    # under attack, a slightly mismatched but trusted oracle is safer than a same-type
    # quarantined oracle.
    if custom.Allow_Cross_Service_Fallback:
        cross = allowed.copy()
        for x in exclude:
            cross[x] = False
        if np.any(cross):
            return cross.astype(bool)

    # Strict fallback 2: choose top-K by dynamic trust, first inside service type.
    # This avoids returning the full original mask, which let attacked nodes re-enter.
    candidates = service.copy()
    for x in exclude:
        candidates[x] = False
    if np.any(candidates):
        idx = np.where(candidates)[0]
    else:
        idx = np.array([i for i in range(env.oracleNum) if i not in exclude], dtype=int)
    if idx.size == 0:
        return np.ones(env.oracleNum, dtype=bool)
    order = idx[np.argsort(-trust[idx])]
    topk = order[: max(int(custom.TopK_Fallback), 1)]
    out = np.zeros(env.oracleNum, dtype=bool)
    out[topk] = True
    return out.astype(bool)


def choose_hcrl_decision(env: SchedulingEnv, models: Dict[str, OptionActorCritic], args, custom, variant: str,
                         request_attrs, defense_state: DefenseState | None, period: int) -> Tuple[int, int, int, str]:
    policy_name = "HCRL-Oracle"
    s_primary = env.getState(request_attrs, policy_name)

    if variant == "HCRL_no_mask":
        primary_mask = _base_service_mask(env, request_attrs)
    elif variant == "HCRL_TAM_v1":
        primary_mask = trust_mask_v1(env, request_attrs, custom, policy_name=policy_name)
    else:
        primary_mask = trust_mask_v2(env, request_attrs, custom, defense_state, period, policy_name=policy_name)

    primary = int(models["primary"].choose_best_action(s_primary, primary_mask))

    s_mode = env.get_hcrl_mode_state(request_attrs, policy_name)
    mode_mask = env.get_hcrl_mode_mask(request_attrs, policy_name, primary)

    mode_names = list(args.HCRL_Mode_Names)
    if variant == "HCRL_TAM_v2" and custom.Safe_Mode_Override:
        trust = _combined_trust(env, defense_state, policy_name)
        primary_risky = trust[primary] < custom.Trust_Recover_Threshold
        if primary_risky:
            for n in ["single_cost", "single_safe"]:
                if n in mode_names:
                    mode_mask[mode_names.index(n)] = False
            if not np.any(mode_mask):
                for n in ["serial_safe", "parallel_safe", "parallel_fast"]:
                    if n in mode_names:
                        mode_mask[mode_names.index(n)] = True
                        break

    mode_idx = int(models["mode"].choose_best_action(s_mode, mode_mask))
    mode_name = mode_names[mode_idx]

    if mode_name.startswith("single"):
        backup = -1
    else:
        if variant == "HCRL_no_mask":
            backup_mask = np.asarray(env.get_backup_action_mask(request_attrs, primary), dtype=bool)
        elif variant == "HCRL_TAM_v1":
            backup_mask = trust_mask_v1(env, request_attrs, custom, policy_name=policy_name, exclude=[primary])
        else:
            backup_mask = trust_mask_v2(env, request_attrs, custom, defense_state, period, policy_name=policy_name, exclude=[primary])
            if custom.Strict_Backup_Filter:
                trust = _combined_trust(env, defense_state, policy_name)
                backup_mask = backup_mask & (trust >= custom.Trust_Action_Threshold)
                backup_mask[primary] = False
                if not np.any(backup_mask):
                    backup_mask = trust_mask_v2(env, request_attrs, custom, defense_state, period, policy_name=policy_name, exclude=[primary])
        backup = int(models["backup"].choose_best_action(s_primary, backup_mask))

    return int(mode_idx), int(primary), int(backup), str(mode_name)


# -----------------------------
# Experiment loop and summaries
# -----------------------------

def run_one(attack: str, variant: str, args, custom, out_dir: Path, seed_offset: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    np.random.seed(args.Seed + seed_offset)
    random.seed(args.Seed + seed_offset)

    env = SchedulingEnv(args)
    env.initial_reputation()
    models = build_hcrl_models(env, args, custom)
    original_validation = env.oracleValidationProbs.copy()
    original_behavior = env.oracleBehaviorProbs.copy()

    defense_state = DefenseState(
        oracle_num=env.oracleNum,
        threshold=custom.Trust_Action_Threshold,
        recover_threshold=custom.Trust_Recover_Threshold,
        dynamic_threshold=custom.Trust_Dynamic_Threshold,
        quarantine_periods=custom.Quarantine_Periods,
        recover_clean_periods=custom.Recover_Clean_Periods,
        dynamic_bad_alpha=custom.Dynamic_Bad_Alpha,
        dynamic_good_alpha=custom.Dynamic_Good_Alpha,
    )

    req_rows = []
    rep_rows = []
    time_period = 1
    for request_c in range(1, min(custom.Max_Requests, args.Request_Num) + 1):
        if request_c % args.Time_Period_Size == 0:
            time_period += 1
            env.update_reputation(env.get_reputation_factors("HCRL-Oracle"), time_period, "HCRL-Oracle")
            env.reset_reputation_factors()
            defense_state.update_period(env, time_period, "HCRL-Oracle")

            rep = env._effective_reputation_vector("HCRL-Oracle")
            dyn = defense_state.dynamic_trust.copy()
            obs = defense_state.trust_observation(env, "HCRL-Oracle")
            for o in custom.Attack_Oracles + custom.Trusted_Reference_Oracles:
                if 0 <= o < env.oracleNum:
                    rep_rows.append({
                        "attack": attack,
                        "variant": variant,
                        "period": time_period,
                        "oracle": int(o),
                        "group": "attacked" if o in custom.Attack_Oracles else "trusted_reference",
                        "effective_reputation": float(rep[o]),
                        "dynamic_trust": float(dyn[o]),
                        "trust_observation": float(obs[o]),
                        "quarantined": bool(time_period < defense_state.quarantine_until[o] or defense_state.was_quarantined[o]),
                        "clean_periods": int(defense_state.clean_periods[o]),
                    })

        finish, request_attrs = env.workload(request_c)
        period = current_period(int(request_attrs[0]), args.Time_Period_Size)
        apply_attack_profile(env, attack, period, custom, original_validation, original_behavior, defense_state, "HCRL-Oracle")

        mode_idx, primary, backup, mode_name = choose_hcrl_decision(env, models, args, custom, variant, request_attrs, defense_state, period)
        feedback = env.feedback_hcrl(request_attrs, mode_idx, primary, backup, "HCRL-Oracle")

        selected = [primary] + ([backup] if backup is not None and backup >= 0 else [])
        attacked_primary = int(primary in custom.Attack_Oracles)
        attacked_backup = int(backup in custom.Attack_Oracles) if backup is not None and backup >= 0 else 0
        attacked_any = int(any(x in custom.Attack_Oracles for x in selected))
        trusted_primary = int(primary in custom.Trusted_Reference_Oracles)
        trusted_backup = int(backup in custom.Trusted_Reference_Oracles) if backup is not None and backup >= 0 else 0
        trusted_any = int(any(x in custom.Trusted_Reference_Oracles for x in selected))
        safe_primary = int(not attacked_primary)

        trust = _combined_trust(env, defense_state, "HCRL-Oracle")
        req_rows.append({
            "attack": attack,
            "variant": variant,
            "request_id": int(request_attrs[0]) + 1,
            "period": int(period),
            "mode_index": int(mode_idx),
            "mode_name": mode_name,
            "primary_oracle": int(primary),
            "backup_oracle": int(backup),
            "attacked_primary": attacked_primary,
            "attacked_backup": attacked_backup,
            "attacked_any": attacked_any,
            "trusted_primary": trusted_primary,
            "trusted_backup": trusted_backup,
            "trusted_any": trusted_any,
            "safe_primary": safe_primary,
            "primary_trust_score": float(trust[primary]),
            "backup_trust_score": float(trust[backup]) if backup is not None and backup >= 0 else np.nan,
            "primary_quarantined": bool(period < defense_state.quarantine_until[primary] or defense_state.was_quarantined[primary]),
            "backup_quarantined": bool(backup >= 0 and (period < defense_state.quarantine_until[backup] or defense_state.was_quarantined[backup])),
            "reward_total": float(feedback.get("total_reward", np.nan)) if isinstance(feedback, dict) else np.nan,
            "primary_reward": float(feedback.get("primary_reward", np.nan)) if isinstance(feedback, dict) else np.nan,
            "backup_reward": float(feedback.get("backup_reward", np.nan)) if isinstance(feedback, dict) else np.nan,
        })

        # Update dynamic trust more frequently for selected nodes after feedback.
        if variant == "HCRL_TAM_v2" and request_c % max(args.Time_Period_Size // 3, 1) == 0:
            defense_state.update_period(env, period, "HCRL-Oracle")

        if finish:
            break

    req_df = pd.DataFrame(req_rows)
    rep_df = pd.DataFrame(rep_rows)
    if req_df.empty:
        raise RuntimeError("No request rows generated")

    period_df = req_df.groupby(["attack", "variant", "period"], as_index=False).agg(
        requests=("request_id", "count"),
        attacked_primary_rate=("attacked_primary", "mean"),
        attacked_backup_rate=("attacked_backup", "mean"),
        attacked_any_rate=("attacked_any", "mean"),
        trusted_primary_rate=("trusted_primary", "mean"),
        trusted_backup_rate=("trusted_backup", "mean"),
        trusted_any_rate=("trusted_any", "mean"),
        safe_primary_rate=("safe_primary", "mean"),
        primary_trust_mean=("primary_trust_score", "mean"),
        backup_trust_mean=("backup_trust_score", "mean"),
        primary_quarantine_rate=("primary_quarantined", "mean"),
        backup_quarantine_rate=("backup_quarantined", "mean"),
    )

    rep_last = rep_df.groupby(["attack", "variant", "group", "oracle"], as_index=False).tail(1) if not rep_df.empty else pd.DataFrame()
    overall = {
        "attack": attack,
        "variant": variant,
        "requests": int(len(req_df)),
        "attacked_primary_rate": float(req_df["attacked_primary"].mean()),
        "attacked_backup_rate": float(req_df["attacked_backup"].mean()),
        "attacked_any_rate": float(req_df["attacked_any"].mean()),
        "trusted_primary_rate": float(req_df["trusted_primary"].mean()),
        "trusted_backup_rate": float(req_df["trusted_backup"].mean()),
        "trusted_any_rate": float(req_df["trusted_any"].mean()),
        "safe_primary_rate": float(req_df["safe_primary"].mean()),
        "primary_trust_mean": float(req_df["primary_trust_score"].mean()),
        "backup_trust_mean": float(req_df["backup_trust_score"].mean(skipna=True)),
    }
    if not rep_last.empty:
        attacked_last = rep_last[rep_last["group"] == "attacked"]
        trusted_last = rep_last[rep_last["group"] == "trusted_reference"]
        overall.update({
            "final_attack_mean_reputation": float(attacked_last["effective_reputation"].mean()) if not attacked_last.empty else np.nan,
            "final_trusted_mean_reputation": float(trusted_last["effective_reputation"].mean()) if not trusted_last.empty else np.nan,
            "final_attack_mean_dynamic_trust": float(attacked_last["dynamic_trust"].mean()) if not attacked_last.empty else np.nan,
            "final_trusted_mean_dynamic_trust": float(trusted_last["dynamic_trust"].mean()) if not trusted_last.empty else np.nan,
            "final_attack_quarantined_count": int(attacked_last["quarantined"].sum()) if not attacked_last.empty else 0,
        })
    return req_df, period_df, rep_df, overall


def save_figures(all_period: pd.DataFrame, all_rep: pd.DataFrame, out_dir: Path, custom):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fmt = custom.Figure_Format

    for (attack, variant), g in all_period.groupby(["attack", "variant"]):
        g = g.sort_values("period")
        plt.figure(figsize=(10, 5))
        plt.plot(g["period"], g["attacked_primary_rate"], label="attacked selected as primary")
        plt.plot(g["period"], g["attacked_any_rate"], label="attacked selected (any)")
        plt.plot(g["period"], g["safe_primary_rate"], label="safe primary")
        plt.plot(g["period"], g["trusted_primary_rate"], label="trusted reference primary")
        plt.xlabel("Time period")
        plt.ylabel("Empirical probability")
        plt.ylim(-0.03, 1.03)
        plt.title(f"{attack} / {variant}: selection probability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{attack}_{variant}_selection_probability.{fmt}", dpi=220)
        plt.close()

    if not all_rep.empty:
        for (attack, variant), g in all_rep.groupby(["attack", "variant"]):
            plt.figure(figsize=(10, 5))
            for (group, oracle), gg in g.groupby(["group", "oracle"]):
                gg = gg.sort_values("period")
                plt.plot(gg["period"], gg["effective_reputation"], label=f"{group}-{oracle}")
            plt.xlabel("Time period")
            plt.ylabel("Effective reputation")
            plt.ylim(-0.03, 1.03)
            plt.title(f"{attack} / {variant}: reputation trajectories")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(fig_dir / f"{attack}_{variant}_reputation.{fmt}", dpi=220)
            plt.close()

            plt.figure(figsize=(10, 5))
            for (group, oracle), gg in g.groupby(["group", "oracle"]):
                gg = gg.sort_values("period")
                plt.plot(gg["period"], gg["dynamic_trust"], label=f"{group}-{oracle}")
            plt.xlabel("Time period")
            plt.ylabel("Dynamic trust score")
            plt.ylim(-0.03, 1.03)
            plt.title(f"{attack} / {variant}: TAM-v2 dynamic trust")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(fig_dir / f"{attack}_{variant}_dynamic_trust.{fmt}", dpi=220)
            plt.close()


def main():
    args, custom = parse_all_args()
    out_root = Path(custom.Output_Dir_Attack)
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root
    run_dir = out_root / f"hcrl_tam_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[HCRL-TAM-v2] Output:", run_dir)
    print("[HCRL-TAM-v2] Attacks:", custom.Attacks)
    print("[HCRL-TAM-v2] Variants:", custom.Variants)
    print("[HCRL-TAM-v2] Attack oracles:", custom.Attack_Oracles)
    print("[HCRL-TAM-v2] Trusted reference:", custom.Trusted_Reference_Oracles)

    all_req, all_period, all_rep, overall_rows = [], [], [], []
    seed_offset = 0
    for attack in custom.Attacks:
        for variant in custom.Variants:
            print(f"\n=== Running {attack} / {variant} ===")
            req_df, period_df, rep_df, overall = run_one(attack, variant, args, custom, run_dir, seed_offset=seed_offset)
            all_req.append(req_df); all_period.append(period_df); all_rep.append(rep_df); overall_rows.append(overall)
            seed_offset += 100
            print(json.dumps(overall, indent=2, ensure_ascii=False))

    req = pd.concat(all_req, ignore_index=True)
    period = pd.concat(all_period, ignore_index=True)
    rep = pd.concat(all_rep, ignore_index=True) if all_rep else pd.DataFrame()
    overall = pd.DataFrame(overall_rows)

    req.to_csv(run_dir / "attack_request_log.csv", index=False, encoding="utf-8-sig")
    period.to_csv(run_dir / "attack_period_summary.csv", index=False, encoding="utf-8-sig")
    rep.to_csv(run_dir / "attack_reputation_trajectory.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(run_dir / "attack_overall_summary.csv", index=False, encoding="utf-8-sig")
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"baseline_args": vars(args), "custom_args": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(custom).items()}}, f, indent=2, ensure_ascii=False, default=str)

    save_figures(period, rep, run_dir, custom)
    print("\nSaved:")
    print("  ", run_dir / "attack_overall_summary.csv")
    print("  ", run_dir / "attack_period_summary.csv")
    print("  ", run_dir / "figures")


if __name__ == "__main__":
    main()
