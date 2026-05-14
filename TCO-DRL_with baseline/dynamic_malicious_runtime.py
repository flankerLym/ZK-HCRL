# -*- coding: utf-8 -*-
"""
Runtime dynamic malicious-oracle training support for TCO-DRL.

This module is intentionally imported by the replacement main.py and monkey-patches
SchedulingEnv at runtime. It does not require manual patch scripts and does not
modify env.py or param_parser.py.

Design:
- The original repo creates fixed malicious oracle ids through Malicious_Oracle_Index.
- During dynamic training, we first use those fixed ids only to estimate a malicious
  feature prototype.
- Then, for each episode / period / request interval, we randomly activate a new
  subset of oracle ids and temporarily project only those ids to the malicious profile.
- All non-active ids, including originally fixed malicious ids, are restored to a
  clean benign-like profile so the agent cannot memorize fixed oracle indices.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np


def extract_dynamic_malicious_args(argv=None):
    """Strip dynamic-malicious CLI args before param_parser.py sees them.

    param_parser.py in the original repo uses parse_args(), so unknown args would
    fail. main.py calls this helper first, then passes the cleaned argv to get_args().
    """
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--Dynamic_Malicious_Training", action="store_true")
    parser.add_argument("--Dynamic_Malicious_Refresh", choices=["episode", "period", "request"], default="episode")
    parser.add_argument("--Dynamic_Malicious_Refresh_Periods", type=int, default=1)
    parser.add_argument("--Dynamic_Malicious_Ratio", type=float, default=-1.0)
    parser.add_argument("--Dynamic_Malicious_Count", type=int, default=-1)
    parser.add_argument("--Dynamic_Malicious_Strategy", choices=["service_balanced", "global"], default="service_balanced")
    parser.add_argument("--Dynamic_Malicious_Profile_Strength", type=float, default=1.0)
    parser.add_argument("--Dynamic_Malicious_Log", action="store_true")
    parser.add_argument("--Dynamic_Malicious_Seed_Offset", type=int, default=7919)
    parser.add_argument("--Disable_Dynamic_Malicious_Curriculum", action="store_true")
    parser.add_argument("--Dynamic_Malicious_Start_Ratio", type=float, default=0.08)
    parser.add_argument("--Dynamic_Malicious_Start_Strength", type=float, default=0.55)
    parser.add_argument("--Dynamic_Malicious_Warmup_Episodes", type=int, default=12)
    parser.add_argument("--Disable_Dynamic_HCRL_Tune", action="store_true")
    parser.add_argument("--Dynamic_HCRL_Tune_Mode", choices=["balanced", "safe", "aggressive"], default="balanced")

    dyn, remaining = parser.parse_known_args(argv[1:])
    return dyn, [argv[0]] + remaining


def attach_dynamic_malicious_args(args, dyn_args):
    for k, v in vars(dyn_args).items():
        setattr(args, k, v)
    return args


def _as_int_list(x):
    if x is None:
        return []
    return [int(i) for i in list(x)]


def install_dynamic_malicious_training(SchedulingEnv):
    """Install monkey-patches once on SchedulingEnv."""
    if getattr(SchedulingEnv, "_dynamic_malicious_patch_installed", False):
        return SchedulingEnv

    orig_init = SchedulingEnv.__init__
    orig_reset = SchedulingEnv.reset
    orig_workload = SchedulingEnv.workload

    def patched_init(self, args):
        orig_init(self, args)
        self._dm_rng = np.random.RandomState(int(getattr(args, "Seed", 6)) + int(getattr(args, "Dynamic_Malicious_Seed_Offset", 7919)))
        _dm_init(self, reason="init")

    def patched_reset(self, args):
        orig_reset(self, args)
        if not hasattr(self, "_dm_rng"):
            self._dm_rng = np.random.RandomState(int(getattr(args, "Seed", 6)) + int(getattr(args, "Dynamic_Malicious_Seed_Offset", 7919)))
        _dm_init(self, reason="episode_reset")

    def patched_workload(self, request_count):
        if bool(getattr(self.args, "Dynamic_Malicious_Training", False)):
            _dm_maybe_refresh(self, trigger="request", key=int(request_count))
        return orig_workload(self, request_count)

    SchedulingEnv.__init__ = patched_init
    SchedulingEnv.reset = patched_reset
    SchedulingEnv.workload = patched_workload

    # Public helper for main.py period-level refresh.
    SchedulingEnv.maybe_refresh_dynamic_malicious = _dm_maybe_refresh

    SchedulingEnv._dynamic_malicious_patch_installed = True
    return SchedulingEnv


def _dm_init(env, reason="init"):
    env.dynamic_malicious_enabled = bool(getattr(env.args, "Dynamic_Malicious_Training", False))
    if not hasattr(env, "dynamic_malicious_generation"):
        env.dynamic_malicious_generation = 0
    env.dynamic_malicious_history = []

    if not env.dynamic_malicious_enabled:
        return

    _dm_build_profiles(env)
    _dm_refresh(env, trigger="episode", key=0, force=True)

    if bool(getattr(env.args, "Dynamic_Malicious_Log", False)):
        print(f"[Dynamic malicious] initialized reason={reason}")


def _dm_build_profiles(env):
    n = int(env.oracleNum)
    all_ids = np.arange(n, dtype=int)
    fixed_mal = np.asarray([i for i in _as_int_list(getattr(env.args, "Malicious_Oracle_Index", [])) if 0 <= i < n], dtype=int)
    fixed_mal_set = set(int(i) for i in fixed_mal.tolist())

    if fixed_mal.size == 0:
        k = max(1, int(round(0.2 * n)))
        fixed_mal = np.argsort(np.asarray(env.oracleValidationProbs, dtype=float))[:k]
        fixed_mal_set = set(int(i) for i in fixed_mal.tolist())

    benign_ids = np.asarray([int(i) for i in all_ids if int(i) not in fixed_mal_set], dtype=int)
    if benign_ids.size == 0:
        benign_ids = all_ids

    env._dm_fixed_malicious_oracles = [int(i) for i in fixed_mal.tolist()]
    env._dm_orig_oracleAcc = np.asarray(env.oracleAcc, dtype=float).copy()
    env._dm_orig_oracleCost = np.asarray(env.oracleCost, dtype=float).copy()
    env._dm_orig_oracleToken = np.asarray(env.oracleToken, dtype=float).copy()
    env._dm_orig_oracleBehaviorProbs = np.asarray(env.oracleBehaviorProbs, dtype=float).copy()
    env._dm_orig_oracleValidationProbs = np.asarray(env.oracleValidationProbs, dtype=float).copy()
    env._dm_orig_oracleFatigueSensitivity = np.asarray(env.oracleFatigueSensitivity, dtype=float).copy()

    def mean_scalar(arr, ids):
        return float(np.asarray(arr, dtype=float)[ids].mean())

    env._dm_mal_proto_acc = mean_scalar(env._dm_orig_oracleAcc, fixed_mal)
    env._dm_mal_proto_cost = mean_scalar(env._dm_orig_oracleCost, fixed_mal)
    env._dm_mal_proto_token = mean_scalar(env._dm_orig_oracleToken, fixed_mal)
    env._dm_mal_proto_validation = mean_scalar(env._dm_orig_oracleValidationProbs, fixed_mal)
    env._dm_mal_proto_fatigue = mean_scalar(env._dm_orig_oracleFatigueSensitivity, fixed_mal)
    b = np.asarray(env._dm_orig_oracleBehaviorProbs[fixed_mal].mean(axis=0), dtype=float)
    env._dm_mal_proto_behavior = b / max(float(b.sum()), 1e-8)

    # Clean baseline: neutralize originally fixed malicious ids by replacing them
    # with same-service benign averages. This avoids oracle-id leakage.
    env._dm_clean_oracleAcc = env._dm_orig_oracleAcc.copy()
    env._dm_clean_oracleCost = env._dm_orig_oracleCost.copy()
    env._dm_clean_oracleToken = env._dm_orig_oracleToken.copy()
    env._dm_clean_oracleBehaviorProbs = env._dm_orig_oracleBehaviorProbs.copy()
    env._dm_clean_oracleValidationProbs = env._dm_orig_oracleValidationProbs.copy()
    env._dm_clean_oracleFatigueSensitivity = env._dm_orig_oracleFatigueSensitivity.copy()

    global_behavior = np.asarray(env._dm_orig_oracleBehaviorProbs[benign_ids].mean(axis=0), dtype=float)
    global_behavior = global_behavior / max(float(global_behavior.sum()), 1e-8)

    for i in fixed_mal.tolist():
        same_type = np.asarray(
            [j for j in benign_ids.tolist() if int(env.oracleTypes[j]) == int(env.oracleTypes[i])],
            dtype=int,
        )
        ids = same_type if same_type.size > 0 else benign_ids
        env._dm_clean_oracleAcc[i] = float(env._dm_orig_oracleAcc[ids].mean())
        env._dm_clean_oracleCost[i] = float(env._dm_orig_oracleCost[ids].mean())
        env._dm_clean_oracleToken[i] = float(env._dm_orig_oracleToken[ids].mean())
        bb = np.asarray(env._dm_orig_oracleBehaviorProbs[ids].mean(axis=0), dtype=float)
        env._dm_clean_oracleBehaviorProbs[i] = bb / max(float(bb.sum()), 1e-8)
        env._dm_clean_oracleValidationProbs[i] = float(env._dm_orig_oracleValidationProbs[ids].mean())
        env._dm_clean_oracleFatigueSensitivity[i] = float(env._dm_orig_oracleFatigueSensitivity[ids].mean())



def _dm_curriculum_progress(env):
    if bool(getattr(env.args, "Disable_Dynamic_Malicious_Curriculum", False)):
        return 1.0
    cur = int(getattr(env.args, "Dynamic_Malicious_Current_Episode", 0))
    warm = max(int(getattr(env.args, "Dynamic_Malicious_Warmup_Episodes", 12)), 1)
    return float(np.clip(cur / float(warm), 0.0, 1.0))


def _dm_effective_ratio(env):
    target = float(getattr(env.args, "Dynamic_Malicious_Ratio", -1.0))
    if target <= 0:
        target = len(getattr(env, "_dm_fixed_malicious_oracles", [])) / max(float(env.oracleNum), 1.0)
    if bool(getattr(env.args, "Disable_Dynamic_Malicious_Curriculum", False)):
        return target
    start = float(getattr(env.args, "Dynamic_Malicious_Start_Ratio", 0.08))
    start = float(np.clip(start, 0.01, max(target, 0.01)))
    p = _dm_curriculum_progress(env)
    return float(start + (target - start) * p)


def _dm_effective_strength(env):
    target = float(np.clip(getattr(env.args, "Dynamic_Malicious_Profile_Strength", 1.0), 0.0, 1.0))
    if bool(getattr(env.args, "Disable_Dynamic_Malicious_Curriculum", False)):
        return target
    start = float(np.clip(getattr(env.args, "Dynamic_Malicious_Start_Strength", 0.55), 0.0, target))
    p = _dm_curriculum_progress(env)
    return float(start + (target - start) * p)


def _dm_attack_count(env):
    n = int(env.oracleNum)
    exact = int(getattr(env.args, "Dynamic_Malicious_Count", -1))
    if exact > 0:
        return int(np.clip(exact, 1, n))
    ratio = _dm_effective_ratio(env)
    return int(np.clip(round(ratio * n), 1, n))


def _dm_sample_ids(env):
    n = int(env.oracleNum)
    k_total = _dm_attack_count(env)
    all_ids = np.arange(n, dtype=int)
    strategy = str(getattr(env.args, "Dynamic_Malicious_Strategy", "service_balanced"))

    if strategy == "global":
        return sorted(int(i) for i in env._dm_rng.choice(all_ids, size=k_total, replace=False).tolist())

    types = np.unique(np.asarray(env.oracleTypes, dtype=int))
    if len(types) == 0:
        return sorted(int(i) for i in env._dm_rng.choice(all_ids, size=k_total, replace=False).tolist())

    active = []
    base = k_total // len(types)
    rem = k_total % len(types)
    order = env._dm_rng.permutation(types)
    quota = {int(t): int(base) for t in types.tolist()}
    for t in order[:rem]:
        quota[int(t)] += 1

    if base == 0:
        quota = {int(t): 0 for t in types.tolist()}
        for t in order[:k_total]:
            quota[int(t)] = 1

    for t in types.tolist():
        q = int(quota[int(t)])
        if q <= 0:
            continue
        cand = np.where(np.asarray(env.oracleTypes, dtype=int) == int(t))[0]
        if cand.size == 0:
            continue
        q = min(q, cand.size)
        active.extend(int(x) for x in env._dm_rng.choice(cand, size=q, replace=False).tolist())

    if len(active) < k_total:
        rem_ids = [int(i) for i in all_ids.tolist() if int(i) not in set(active)]
        if rem_ids:
            extra = env._dm_rng.choice(np.asarray(rem_ids, dtype=int), size=min(k_total - len(active), len(rem_ids)), replace=False)
            active.extend(int(x) for x in extra.tolist())

    return sorted(active[:k_total])


def _dm_apply_profile(env, active_ids: List[int]):
    active_ids = sorted(int(i) for i in active_ids if 0 <= int(i) < int(env.oracleNum))
    s = _dm_effective_strength(env)

    # Restore clean community first.
    env.oracleAcc = env._dm_clean_oracleAcc.copy()
    env.oracleCost = env._dm_clean_oracleCost.copy()
    env.oracleToken = env._dm_clean_oracleToken.copy()
    env.oracleBehaviorProbs = env._dm_clean_oracleBehaviorProbs.copy()
    env.oracleValidationProbs = env._dm_clean_oracleValidationProbs.copy()
    env.oracleFatigueSensitivity = env._dm_clean_oracleFatigueSensitivity.copy()

    # Project selected active attackers to malicious prototype.
    for i in active_ids:
        env.oracleAcc[i] = (1.0 - s) * env.oracleAcc[i] + s * env._dm_mal_proto_acc
        env.oracleCost[i] = (1.0 - s) * env.oracleCost[i] + s * env._dm_mal_proto_cost
        env.oracleToken[i] = (1.0 - s) * env.oracleToken[i] + s * env._dm_mal_proto_token
        env.oracleValidationProbs[i] = (1.0 - s) * env.oracleValidationProbs[i] + s * env._dm_mal_proto_validation
        env.oracleFatigueSensitivity[i] = (1.0 - s) * env.oracleFatigueSensitivity[i] + s * env._dm_mal_proto_fatigue
        bb = (1.0 - s) * env.oracleBehaviorProbs[i] + s * env._dm_mal_proto_behavior
        env.oracleBehaviorProbs[i] = bb / max(float(bb.sum()), 1e-8)

    env.malicious_oracles = active_ids
    env.active_malicious_oracles = active_ids
    env.args.Malicious_Oracle_Index = list(active_ids)


def _dm_refresh(env, trigger="episode", key=0, force=False):
    if not bool(getattr(env.args, "Dynamic_Malicious_Training", False)):
        return
    active = _dm_sample_ids(env)
    _dm_apply_profile(env, active)
    env.dynamic_malicious_generation = int(getattr(env, "dynamic_malicious_generation", 0)) + 1
    rec = {
        "generation": int(env.dynamic_malicious_generation),
        "trigger": str(trigger),
        "key": int(key) if key is not None else -1,
        "active_malicious_oracles": list(active),
        "effective_ratio": float(_dm_effective_ratio(env)),
        "effective_strength": float(_dm_effective_strength(env)),
    }
    if not hasattr(env, "dynamic_malicious_history"):
        env.dynamic_malicious_history = []
    env.dynamic_malicious_history.append(rec)
    if bool(getattr(env.args, "Dynamic_Malicious_Log", False)):
        print(f"[Dynamic malicious] generation={rec['generation']} trigger={trigger} key={key} "
              f"ratio={rec['effective_ratio']:.3f} strength={rec['effective_strength']:.3f} active={active}")


def _dm_maybe_refresh(env, trigger="period", time_period=None, request_id=None, key=None):
    if not bool(getattr(env.args, "Dynamic_Malicious_Training", False)):
        return
    refresh = str(getattr(env.args, "Dynamic_Malicious_Refresh", "episode"))
    if refresh != str(trigger):
        return

    interval = max(int(getattr(env.args, "Dynamic_Malicious_Refresh_Periods", 1)), 1)
    if trigger == "period":
        k = int(time_period if time_period is not None else (key if key is not None else 0))
    elif trigger == "request":
        k = int(request_id if request_id is not None else (key if key is not None else 0))
    else:
        k = int(key if key is not None else 0)

    if trigger in {"period", "request"} and (k <= 0 or k % interval != 0):
        return
    _dm_refresh(env, trigger=trigger, key=k, force=True)
