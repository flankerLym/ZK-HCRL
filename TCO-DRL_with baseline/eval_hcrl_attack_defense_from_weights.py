
"""
Evaluate saved HCRL weights under dynamic-random ME / OOA / OSA malicious-oracle feature engineering.

Replace:
    TCO-DRL_with baseline/eval_hcrl_attack_defense_from_weights.py

Main goal:
    Use the saved HCRL actor-critic weights from TCO-DRL/weight/audit_hcrl,
    simulate dynamic random malicious profiles (ME, OOA, OSA), and measure whether HCRL
    identifies / avoids / penalizes malicious oracles through reputation,
    validation success, behavior risk, audit risk, latency, and cost.

Run examples:
    # From repository root:
    python "TCO-DRL_with baseline/eval_hcrl_attack_defense_from_weights.py"

    # Or from TCO-DRL_with baseline:
    python eval_hcrl_attack_defense_from_weights.py

    # Explicit weight path:
    python eval_hcrl_attack_defense_from_weights.py --Weight_Dir ../weight/audit_hcrl

Outputs:
    attack_request_log.csv
    attack_period_summary.csv
    attack_reputation_trajectory.csv
    attack_oracle_feature_log.csv
    attack_detection_summary.csv
    attack_overall_summary.csv
    run_config.json
    figures/*.png
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from utils import get_args
try:
    # Needed when the saved HCRL weights were trained with a different oracle
    # community size than the parser defaults. We regenerate the oracle arrays
    # after inferring Oracle_Num / State_Mode from the checkpoint shapes.
    from param_parser import _generate_oracle_community
except Exception:
    _generate_oracle_community = None

from env import SchedulingEnv
from model import OptionActorCritic


POLICY_NAME = "HCRL-Oracle"
DEFAULT_MODES = ["single_cost", "single_safe", "serial_safe", "parallel_fast", "parallel_safe"]


def ensure_attr(args, name: str, value):
    if not hasattr(args, name):
        setattr(args, name, value)
    return getattr(args, name)


def as_int_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, str):
        if not x.strip():
            return []
        return [int(v) for v in x.replace(",", " ").split()]
    return [int(v) for v in list(x)]


def as_float_array(x, dtype=float) -> np.ndarray:
    return np.asarray(list(x), dtype=dtype)


def normalize_probs(p: Sequence[float]) -> List[float]:
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return (arr / s).tolist()


def get_script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd().resolve()


def resolve_weight_dir(path_like: str) -> Path:
    """Resolve weight/audit_hcrl robustly from repo root or script folder."""
    if path_like and path_like.lower() not in {"auto", "default"}:
        p = Path(path_like)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    script_dir = get_script_dir()
    candidates = [
        Path.cwd() / "weight" / "audit_hcrl",
        Path.cwd() / "TCO-DRL" / "weight" / "audit_hcrl",
        Path.cwd().parent / "weight" / "audit_hcrl",
        script_dir.parent / "weight" / "audit_hcrl",
        script_dir / ".." / "weight" / "audit_hcrl",
    ]
    for c in candidates:
        c = c.resolve()
        if c.exists():
            return c
    return (script_dir.parent / "weight" / "audit_hcrl").resolve()


def find_weight_file(weight_dir: Path, role: str) -> Path:
    """Find HCRL_Mode.npz / HCRL_Primary.npz / HCRL_Backup.npz with fallbacks."""
    role = role.lower()
    preferred = {
        "mode": ["HCRL_Mode.npz", "HCRL_Mode_OptionAC.npz", "Mode.npz"],
        "primary": ["HCRL_Primary.npz", "HCRL_Primary_OptionAC.npz", "Primary.npz"],
        "backup": ["HCRL_Backup.npz", "HCRL_Backup_OptionAC.npz", "Backup.npz"],
    }[role]

    for name in preferred:
        p = weight_dir / name
        if p.exists():
            return p

    patterns = {
        "mode": ["*Mode*.npz", "*mode*.npz"],
        "primary": ["*Primary*.npz", "*primary*.npz"],
        "backup": ["*Backup*.npz", "*backup*.npz"],
    }[role]

    for pat in patterns:
        hits = sorted(weight_dir.glob(pat))
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"Cannot find {role} HCRL weight in {weight_dir}. "
        f"Expected one of: {preferred}"
    )



def read_npz_shape(path: Path) -> Dict:
    data = np.load(str(path), allow_pickle=True)
    if "W1" not in data.files:
        raise ValueError(f"{path} has no W1; keys={data.files}")

    W1 = np.asarray(data["W1"])
    n_features = int(np.asarray(data["n_features"]).item()) if "n_features" in data.files else int(W1.shape[0])
    hidden_dim = int(W1.shape[1])

    if "Wp" in data.files:
        n_actions = int(np.asarray(data["Wp"]).shape[1])
    elif "W2" in data.files:
        n_actions = int(np.asarray(data["W2"]).shape[1])
    elif "Wa" in data.files:
        n_actions = int(np.asarray(data["Wa"]).shape[1])
    elif "n_actions" in data.files:
        n_actions = int(np.asarray(data["n_actions"]).item())
    else:
        n_actions = None

    metadata = ""
    if "metadata" in data.files:
        try:
            metadata = str(np.asarray(data["metadata"]).item())
        except Exception:
            metadata = str(data["metadata"])

    return {
        "path": str(path),
        "input_dim": n_features,
        "w1_shape": tuple(W1.shape),
        "hidden_dim": hidden_dim,
        "n_actions": n_actions,
        "metadata": metadata,
        "keys": list(data.files),
    }


def inspect_hcrl_weight_shapes(weight_dir: Path) -> Dict[str, Dict]:
    shapes = {}
    for role in ["mode", "primary", "backup"]:
        path = find_weight_file(weight_dir, role)
        shapes[role] = read_npz_shape(path)
    return shapes


def _infer_state_mode_and_oracle_num(primary_dim: int, action_num: int | None) -> Tuple[str, int]:
    candidates = []

    if primary_dim is not None:
        if primary_dim >= 3 and (primary_dim - 3) % 12 == 0:
            n = (primary_dim - 3) // 12
            if n > 0:
                candidates.append(("enhanced", int(n)))
        if primary_dim >= 1 and (primary_dim - 1) % 2 == 0:
            n = (primary_dim - 1) // 2
            if n > 0:
                candidates.append(("original", int(n)))

    if action_num is not None and candidates:
        exact = [c for c in candidates if c[1] == int(action_num)]
        if exact:
            # Prefer enhanced when both formulas accidentally fit.
            exact = sorted(exact, key=lambda x: 0 if x[0] == "enhanced" else 1)
            return exact[0]

    enhanced = [c for c in candidates if c[0] == "enhanced"]
    if enhanced:
        return enhanced[0]
    if candidates:
        return candidates[0]

    if action_num is not None:
        return "enhanced", int(action_num)

    raise ValueError(
        f"Cannot infer State_Mode / Oracle_Num from checkpoint primary_dim={primary_dim}, "
        f"action_num={action_num}."
    )


def align_args_to_hcrl_weights(args, weight_shapes: Dict[str, Dict], custom):
    """Make env state/action dimensions match saved HCRL checkpoints.

    Your error:
        HCRL_Mode W1 checkpoint (373, 64) != model (41, 64)

    means the saved mode policy expects:
        373 = enhanced primary state 363 + 10 mode-summary features
        363 = 3 + 12 * 30 oracle features

    while the current parser defaults produced:
        41 = original primary state 31 + 10
        31 = 1 + 2 * 15 oracle features

    This function fixes that automatically by reading the NPZ shapes before
    constructing SchedulingEnv.
    """
    mode_dim = int(weight_shapes["mode"]["input_dim"])
    primary_dim = int(weight_shapes["primary"]["input_dim"])
    backup_dim = int(weight_shapes["backup"]["input_dim"])
    primary_actions = weight_shapes["primary"].get("n_actions")
    backup_actions = weight_shapes["backup"].get("n_actions")

    if primary_dim != backup_dim:
        raise ValueError(
            f"Primary/Backup input dims differ: primary={primary_dim}, backup={backup_dim}. "
            "Please check that the three NPZ files come from the same training run."
        )

    if mode_dim != primary_dim + 10:
        print(
            f"[warn] HCRL_Mode input_dim={mode_dim}, Primary input_dim={primary_dim}; "
            "expected mode_dim = primary_dim + 10 in this repo. I will still align to "
            "the primary checkpoint and load strictly afterwards."
        )

    action_num = primary_actions or backup_actions
    if primary_actions is not None and backup_actions is not None and int(primary_actions) != int(backup_actions):
        raise ValueError(
            f"Primary/Backup action dims differ: primary={primary_actions}, backup={backup_actions}. "
            "Please check that the checkpoint files match."
        )

    inferred_state_mode, inferred_oracle_num = _infer_state_mode_and_oracle_num(primary_dim, action_num)

    # Allow explicit override when the user knows the training community layout.
    override_service = int(getattr(custom, "Weight_Service_Type_Num", 0) or 0)
    override_per_type = int(getattr(custom, "Weight_Oracles_Per_Type", 0) or 0)

    old_state = getattr(args, "State_Mode", None)
    old_oracle_num = int(getattr(args, "Oracle_Num", 0))
    old_service = int(getattr(args, "Service_Type_Num", 1))
    old_per_type = int(getattr(args, "Oracles_Per_Type", 1))

    args.State_Mode = inferred_state_mode
    args.Oracle_Num = int(action_num or inferred_oracle_num)

    if override_service > 0 and override_per_type > 0:
        args.Service_Type_Num = override_service
        args.Oracles_Per_Type = override_per_type
    elif old_per_type > 0 and args.Oracle_Num % old_per_type == 0:
        # For your current case this maps 30 oracles to 6 service types * 5 per type.
        args.Oracles_Per_Type = old_per_type
        args.Service_Type_Num = args.Oracle_Num // old_per_type
    elif old_service > 0 and args.Oracle_Num % old_service == 0:
        args.Service_Type_Num = old_service
        args.Oracles_Per_Type = args.Oracle_Num // old_service
    else:
        args.Service_Type_Num = 1
        args.Oracles_Per_Type = args.Oracle_Num

    if args.Service_Type_Num * args.Oracles_Per_Type != args.Oracle_Num:
        raise ValueError(
            f"Inferred Oracle_Num={args.Oracle_Num}, but Service_Type_Num * Oracles_Per_Type = "
            f"{args.Service_Type_Num} * {args.Oracles_Per_Type}. Pass "
            "--Weight_Service_Type_Num and --Weight_Oracles_Per_Type explicitly."
        )

    # Regenerate Oracle_Type, Cost, Acc, Tokens, Behavior, Validation, and malicious/trusted indices.
    if _generate_oracle_community is not None:
        _generate_oracle_community(args)
    else:
        raise RuntimeError(
            "Cannot import param_parser._generate_oracle_community. "
            "Please run the script inside TCO-DRL_with baseline, or pass the same "
            "oracle lists manually in the parser."
        )

    # Hidden sizes must also match checkpoint W1 second dimension.
    args.HCRL_Mode_Hidden = int(weight_shapes["mode"]["hidden_dim"])
    args.HCRL_Primary_Hidden = int(weight_shapes["primary"]["hidden_dim"])
    args.HCRL_Backup_Hidden = int(weight_shapes["backup"]["hidden_dim"])
    args.Dqn_hidden = int(args.HCRL_Primary_Hidden)

    mode_actions = weight_shapes["mode"].get("n_actions")
    if mode_actions is not None and len(getattr(args, "HCRL_Mode_Names", DEFAULT_MODES)) != int(mode_actions):
        if int(mode_actions) <= len(DEFAULT_MODES):
            args.HCRL_Mode_Names = DEFAULT_MODES[:int(mode_actions)]
        else:
            args.HCRL_Mode_Names = DEFAULT_MODES + [f"mode_{i}" for i in range(len(DEFAULT_MODES), int(mode_actions))]

    print(
        "[weight-shape alignment] "
        f"State_Mode {old_state}->{args.State_Mode}; "
        f"Oracle_Num {old_oracle_num}->{args.Oracle_Num}; "
        f"Service_Type_Num {old_service}->{args.Service_Type_Num}; "
        f"Oracles_Per_Type {old_per_type}->{args.Oracles_Per_Type}; "
        f"mode_dim={mode_dim}, primary_dim={primary_dim}, actions={action_num}; "
        f"hidden(mode/primary/backup)="
        f"{args.HCRL_Mode_Hidden}/{args.HCRL_Primary_Hidden}/{args.HCRL_Backup_Hidden}"
    )

    return args

def load_npz_checkpoint(model, path: Path, strict: bool = True):
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(str(path), allow_pickle=True)
    loaded = []

    for name in ["W1", "b1", "Wp", "bp", "Wv", "bv", "W2", "b2", "Wa", "ba"]:
        if name not in data.files:
            continue
        arr = np.asarray(data[name], dtype=np.float32)

        target_name = name
        if name in ["W2", "Wa"] and hasattr(model, "Wp") and not hasattr(model, name):
            target_name = "Wp"
        elif name in ["b2", "ba"] and hasattr(model, "bp") and not hasattr(model, name):
            target_name = "bp"

        if not hasattr(model, target_name):
            continue

        cur = getattr(model, target_name)
        if strict and hasattr(cur, "shape") and tuple(cur.shape) != tuple(arr.shape):
            raise ValueError(
                f"Shape mismatch for {path.name}:{name}->{target_name}: "
                f"checkpoint {arr.shape} != model {cur.shape}. "
                f"Please pass the same training args, especially --State_Mode, "
                f"--Dqn_hidden, --Service_Type_Num, --Oracles_Per_Type."
            )

        if hasattr(cur, "shape") and tuple(cur.shape) == tuple(arr.shape):
            setattr(model, target_name, arr)
            loaded.append(f"{name}->{target_name}")

    if not loaded:
        raise ValueError(f"No compatible arrays in {path}; keys={data.files}")

    return loaded


def configure_hcrl_eval_args(args):
    args.Baselines = [POLICY_NAME]
    args.Baseline_num = 1

    ensure_attr(args, "HCRL_Mode_Names", DEFAULT_MODES)
    ensure_attr(args, "Use_Audit_Reputation", True)
    ensure_attr(args, "Success_Mode", "validation_aware")
    ensure_attr(args, "Action_Mask_Mode", "type")
    ensure_attr(args, "Reward_Clip", 3.0)
    ensure_attr(args, "Dqn_hidden", 128)
    ensure_attr(args, "Dqn_memory_size", 6000)
    ensure_attr(args, "Dqn_batch_size", 64)
    ensure_attr(args, "HCRL_Mode_lr", 0.001)
    ensure_attr(args, "HCRL_lr", 0.001)
    ensure_attr(args, "HCRL_AC_Entropy", 0.01)
    ensure_attr(args, "HCRL_AC_Value_Coef", 0.5)

    ensure_attr(args, "Attack_Mode", "NONE")
    ensure_attr(args, "Attack_Oracles", as_int_list(getattr(args, "Malicious_Oracle_Index", [])))
    ensure_attr(args, "Trusted_Reference_Oracles", as_int_list(getattr(args, "Trusted_Oracle_Index", [])))
    return args


def build_hcrl_models(args, env, weight_dir: Path) -> Dict[str, OptionActorCritic]:
    mode_dim = int(env.s_features + env.mode_extra_features)

    mode_hidden = int(getattr(args, "HCRL_Mode_Hidden", max(64, int(args.Dqn_hidden) // 2)))
    primary_hidden = int(getattr(args, "HCRL_Primary_Hidden", int(args.Dqn_hidden)))
    backup_hidden = int(getattr(args, "HCRL_Backup_Hidden", int(args.Dqn_hidden)))

    models = {
        "mode": OptionActorCritic(
            len(getattr(args, "HCRL_Mode_Names", DEFAULT_MODES)),
            mode_dim,
            hidden_units=mode_hidden,
            scope="HCRL_Mode_OptionAC",
            learning_rate=float(args.HCRL_Mode_lr),
            memory_size=int(args.Dqn_memory_size),
            batch_size=int(args.Dqn_batch_size),
            entropy_coef=float(args.HCRL_AC_Entropy),
            value_coef=float(args.HCRL_AC_Value_Coef),
            reward_clip=float(args.Reward_Clip),
            seed=int(args.Seed) + 3031,
        ),
        "primary": OptionActorCritic(
            env.actionNum,
            env.s_features,
            hidden_units=primary_hidden,
            scope="HCRL_Primary_OptionAC",
            learning_rate=float(args.HCRL_lr),
            memory_size=int(args.Dqn_memory_size),
            batch_size=int(args.Dqn_batch_size),
            entropy_coef=float(args.HCRL_AC_Entropy),
            value_coef=float(args.HCRL_AC_Value_Coef),
            reward_clip=float(args.Reward_Clip),
            seed=int(args.Seed) + 4049,
        ),
        "backup": OptionActorCritic(
            env.actionNum,
            env.s_features,
            hidden_units=backup_hidden,
            scope="HCRL_Backup_OptionAC",
            learning_rate=float(args.HCRL_lr),
            memory_size=int(args.Dqn_memory_size),
            batch_size=int(args.Dqn_batch_size),
            entropy_coef=float(args.HCRL_AC_Entropy),
            value_coef=float(args.HCRL_AC_Value_Coef),
            reward_clip=float(args.Reward_Clip),
            seed=int(args.Seed) + 5051,
        ),
    }

    loaded = {
        "mode": load_npz_checkpoint(models["mode"], find_weight_file(weight_dir, "mode")),
        "primary": load_npz_checkpoint(models["primary"], find_weight_file(weight_dir, "primary")),
        "backup": load_npz_checkpoint(models["backup"], find_weight_file(weight_dir, "backup")),
    }

    print("[weights] loaded from:", weight_dir)
    for k, v in loaded.items():
        print(f"[weights] {k}: {v}")

    return models


def default_attack_oracles(args) -> List[int]:
    malicious = as_int_list(getattr(args, "Malicious_Oracle_Index", []))
    if malicious:
        return malicious

    service_type_num = int(getattr(args, "Service_Type_Num", 3))
    per_type = int(getattr(args, "Oracles_Per_Type", 5))
    return [t * per_type for t in range(service_type_num)]


def _project_arrays_for_attack(cost, acc, tokens, val, beh, fatigue, attack_oracles, attack_name: str, custom):
    """Project selected oracle feature arrays toward the training-time malicious prototype."""
    attack_name = attack_name.upper().strip()
    for i in attack_oracles:
        if i < 0 or i >= len(cost):
            continue

        if attack_name == "ME":
            cost[i] = max(0.01, cost[i] * float(custom.ME_Cost_Scale))
            acc[i] = max(1e-4, acc[i] * float(custom.ME_Acc_Scale))
            tokens[i] = max(1.0, tokens[i] * float(custom.ME_Token_Scale))
            val[i] = min(float(val[i]), float(custom.ME_Validation_Prob))
            beh[i] = normalize_probs([
                float(custom.ME_Behavior_P0),
                float(custom.ME_Behavior_P1),
                float(custom.ME_Behavior_P5),
                float(custom.ME_Behavior_P100),
            ])
            fatigue[i] = max(float(fatigue[i]), float(custom.ME_Fatigue))

        elif attack_name == "OOA":
            cost[i] = max(0.01, cost[i] * float(custom.OOA_Cost_Scale))
            acc[i] = max(1e-4, acc[i] * float(custom.OOA_Acc_Scale))
            tokens[i] = max(1.0, tokens[i] * float(custom.OOA_Token_Scale))
            # OOA uses clean validation prior before it switches on. Runtime behavior below
            # turns it into a malicious executor after the local warm-up interval.
            val[i] = max(float(val[i]), float(custom.OOA_Off_Validation_Prob))
            beh[i] = normalize_probs([
                float(custom.OOA_Off_Behavior_P0),
                float(custom.OOA_Off_Behavior_P1),
                float(custom.OOA_Off_Behavior_P5),
                float(custom.OOA_Off_Behavior_P100),
            ])
            fatigue[i] = max(float(fatigue[i]), float(custom.OOA_Fatigue))

        elif attack_name == "OSA":
            cost[i] = max(0.01, cost[i] * float(custom.OSA_Cost_Scale))
            acc[i] = max(1e-4, acc[i] * float(custom.OSA_Acc_Scale))
            tokens[i] = max(1.0, tokens[i] * float(custom.OSA_Token_Scale))
            val[i] = min(max(float(val[i]), float(custom.OSA_Base_Validation_Prob)), 0.99)
            beh[i] = normalize_probs([
                float(custom.OSA_Base_Behavior_P0),
                float(custom.OSA_Base_Behavior_P1),
                float(custom.OSA_Base_Behavior_P5),
                float(custom.OSA_Base_Behavior_P100),
            ])
            fatigue[i] = max(float(fatigue[i]), float(custom.OSA_Fatigue))
    return cost, acc, tokens, val, beh, fatigue


def apply_attack_feature_engineering(args, attack_name: str, custom) -> Tuple[object, Dict]:
    """
    Experiment C default:
    - Do NOT bind malicious behavior to fixed oracle IDs.
    - Keep the original oracle community as the clean base distribution.
    - Dynamic random attackers are sampled at runtime and projected to the
      training-time malicious prototype only during their active attack window.
    """
    args = copy.deepcopy(args)
    attack_name = attack_name.upper().strip()
    if attack_name not in {"ME", "OOA", "OSA"}:
        raise ValueError(f"Unknown attack: {attack_name}. Use ME, OOA, OSA.")

    configure_hcrl_eval_args(args)

    trusted_reference = as_int_list(getattr(args, "Trusted_Oracle_Index", []))
    args.Attack_Mode = attack_name
    args.Trusted_Reference_Oracles = trusted_reference

    if int(getattr(custom, "Dynamic_Attack", 1)) == 1:
        # No fixed malicious oracle IDs. The runtime scheduler samples active
        # attackers randomly every attack window.
        args.Attack_Oracles = []
        feature_info = {
            "attack": attack_name,
            "dynamic_attack": 1,
            "dynamic_attack_ratio": float(custom.Dynamic_Attack_Ratio),
            "dynamic_attack_count": int(custom.Dynamic_Attack_Count),
            "dynamic_refresh_periods": int(custom.Dynamic_Attack_Refresh_Periods),
            "dynamic_projection": int(custom.Dynamic_Projection),
            "trusted_reference_oracles": trusted_reference,
            "note": "Experiment C: active malicious oracles are sampled randomly at runtime; no fixed malicious positions are used.",
        }
        return args, feature_info

    # Static fallback, mainly for ablation/debugging.
    attack_oracles = as_int_list(getattr(custom, "Attack_Oracles", None))
    if not attack_oracles:
        attack_oracles = default_attack_oracles(args)
    args.Attack_Oracles = attack_oracles

    cost = as_float_array(args.Oracle_Cost)
    acc = as_float_array(args.Oracle_Acc)
    tokens = as_float_array(args.Oracle_Tokens)
    val = as_float_array(args.Oracle_Validation_Probs)
    beh = np.asarray(args.Oracle_Behavior_Probs, dtype=float)
    fatigue = as_float_array(getattr(args, "Oracle_Fatigue_Sensitivity", [0.0] * int(args.Oracle_Num)))

    max_cost = max(float(np.max(cost)), 1e-8)
    max_token = max(float(np.max(tokens)), 1e-8)
    cost, acc, tokens, val, beh, fatigue = _project_arrays_for_attack(
        cost, acc, tokens, val, beh, fatigue, attack_oracles, attack_name, custom
    )

    args.Oracle_Cost = cost.tolist()
    args.Oracle_Acc = acc.tolist()
    args.Oracle_Tokens = tokens.tolist()
    args.Oracle_Validation_Probs = val.tolist()
    args.Oracle_Behavior_Probs = beh.tolist()
    args.Oracle_Fatigue_Sensitivity = fatigue.tolist()

    feature_info = {
        "attack": attack_name,
        "dynamic_attack": 0,
        "attack_oracles": attack_oracles,
        "trusted_reference_oracles": trusted_reference,
        "cost_norm": {int(i): float(cost[i] / max_cost) for i in attack_oracles if 0 <= i < len(cost)},
        "token_norm": {int(i): float(tokens[i] / max_token) for i in attack_oracles if 0 <= i < len(tokens)},
        "validation_prior": {int(i): float(val[i]) for i in attack_oracles if 0 <= i < len(val)},
        "behavior_probs": {int(i): [float(x) for x in beh[i]] for i in attack_oracles if 0 <= i < len(beh)},
    }
    return args, feature_info


def _maybe_set_env_array(env, names: Sequence[str], value):
    arr = np.asarray(value).copy()
    for name in names:
        if hasattr(env, name):
            try:
                current = getattr(env, name)
                if hasattr(current, "shape") and tuple(np.asarray(current).shape) == tuple(arr.shape):
                    current[...] = arr
                else:
                    setattr(env, name, arr.copy())
            except Exception:
                setattr(env, name, arr.copy())


def ensure_dynamic_attack_state(env: SchedulingEnv, custom):
    if hasattr(env, "_dynamic_attack_ready"):
        return

    seed = int(custom.Dynamic_Attack_Seed)
    if seed < 0:
        seed = int(getattr(env.args, "Seed", 1)) + 7919
    env.dynamic_attack_rng = np.random.RandomState(seed)

    env.base_oracleCost = np.asarray(env.oracleCost, dtype=float).copy()
    env.base_oracleAcc = np.asarray(env.oracleAcc, dtype=float).copy()
    env.base_oracleToken = np.asarray(env.oracleToken, dtype=float).copy()
    env.base_oracleValidationProbs = np.asarray(env.oracleValidationProbs, dtype=float).copy()

    if hasattr(env, "oracleBehaviorProbs"):
        env.base_oracleBehaviorProbs = np.asarray(env.oracleBehaviorProbs, dtype=float).copy()
    elif hasattr(env, "oracleBehavior_Probs"):
        env.base_oracleBehaviorProbs = np.asarray(env.oracleBehavior_Probs, dtype=float).copy()
    else:
        env.base_oracleBehaviorProbs = np.asarray(env.args.Oracle_Behavior_Probs, dtype=float).copy()

    if hasattr(env, "oracleFatigueSensitivity"):
        env.base_oracleFatigueSensitivity = np.asarray(env.oracleFatigueSensitivity, dtype=float).copy()
    elif hasattr(env, "oracleFatigue"):
        env.base_oracleFatigueSensitivity = np.asarray(env.oracleFatigue, dtype=float).copy()
    else:
        env.base_oracleFatigueSensitivity = np.asarray(
            getattr(env.args, "Oracle_Fatigue_Sensitivity", [0.0] * int(env.oracleNum)),
            dtype=float,
        ).copy()

    env.static_malicious_oracles_from_parser = set(getattr(env, "malicious_oracles", set()))
    env.current_attack_set = set()
    env.ever_attack_set = set()
    env.dynamic_attack_window_id = None
    env.dynamic_attack_generation = 0
    env.dynamic_attack_window_start_request = 0
    env.dynamic_attack_history = []
    env._dynamic_attack_ready = True


def _restore_clean_oracle_features(env: SchedulingEnv):
    _maybe_set_env_array(env, ["oracleCost"], env.base_oracleCost)
    _maybe_set_env_array(env, ["oracleAcc"], env.base_oracleAcc)
    _maybe_set_env_array(env, ["oracleToken"], env.base_oracleToken)
    _maybe_set_env_array(env, ["oracleValidationProbs"], env.base_oracleValidationProbs)
    _maybe_set_env_array(env, ["oracleBehaviorProbs", "oracleBehavior_Probs"], env.base_oracleBehaviorProbs)
    _maybe_set_env_array(env, ["oracleFatigueSensitivity", "oracleFatigue"], env.base_oracleFatigueSensitivity)

    env.args.Oracle_Cost = np.asarray(env.oracleCost, dtype=float).tolist()
    env.args.Oracle_Acc = np.asarray(env.oracleAcc, dtype=float).tolist()
    env.args.Oracle_Tokens = np.asarray(env.oracleToken, dtype=float).tolist()
    env.args.Oracle_Validation_Probs = np.asarray(env.oracleValidationProbs, dtype=float).tolist()
    env.args.Oracle_Behavior_Probs = np.asarray(env.base_oracleBehaviorProbs, dtype=float).tolist()
    env.args.Oracle_Fatigue_Sensitivity = np.asarray(env.base_oracleFatigueSensitivity, dtype=float).tolist()


def apply_dynamic_projection_to_active_set(env: SchedulingEnv, attack_name: str, custom):
    """
    Restore all oracles to clean features, then project only the currently active
    random attackers toward the training-time malicious prototype.
    """
    ensure_dynamic_attack_state(env, custom)
    _restore_clean_oracle_features(env)

    active = sorted(list(getattr(env, "current_attack_set", set())))
    if not active or int(getattr(custom, "Dynamic_Projection", 1)) == 0:
        env.malicious_oracles = set(active)
        env.args.Attack_Oracles = active
        return

    cost = np.asarray(env.oracleCost, dtype=float).copy()
    acc = np.asarray(env.oracleAcc, dtype=float).copy()
    tokens = np.asarray(env.oracleToken, dtype=float).copy()
    val = np.asarray(env.oracleValidationProbs, dtype=float).copy()
    beh = np.asarray(env.base_oracleBehaviorProbs, dtype=float).copy()
    fatigue = np.asarray(env.base_oracleFatigueSensitivity, dtype=float).copy()

    cost, acc, tokens, val, beh, fatigue = _project_arrays_for_attack(
        cost, acc, tokens, val, beh, fatigue, active, attack_name, custom
    )

    _maybe_set_env_array(env, ["oracleCost"], cost)
    _maybe_set_env_array(env, ["oracleAcc"], acc)
    _maybe_set_env_array(env, ["oracleToken"], tokens)
    _maybe_set_env_array(env, ["oracleValidationProbs"], val)
    _maybe_set_env_array(env, ["oracleBehaviorProbs", "oracleBehavior_Probs"], beh)
    _maybe_set_env_array(env, ["oracleFatigueSensitivity", "oracleFatigue"], fatigue)

    env.args.Oracle_Cost = cost.tolist()
    env.args.Oracle_Acc = acc.tolist()
    env.args.Oracle_Tokens = tokens.tolist()
    env.args.Oracle_Validation_Probs = val.tolist()
    env.args.Oracle_Behavior_Probs = beh.tolist()
    env.args.Oracle_Fatigue_Sensitivity = fatigue.tolist()

    # Important: replace the parser's fixed malicious set with the current
    # random active set, so the environment's internal reward/audit logic does
    # not leak fixed malicious positions.
    env.malicious_oracles = set(active)
    env.args.Attack_Oracles = active


def update_dynamic_attack_set(env: SchedulingEnv, request_attrs, time_period: int, custom, attack_name: str):
    """Randomly resample active attackers at the beginning of each attack window."""
    ensure_dynamic_attack_state(env, custom)

    if int(getattr(custom, "Dynamic_Attack", 1)) == 0:
        env.current_attack_set = set(as_int_list(getattr(env.args, "Attack_Oracles", [])))
        env.ever_attack_set.update(env.current_attack_set)
        apply_dynamic_projection_to_active_set(env, attack_name, custom)
        return env.current_attack_set

    refresh_periods = max(1, int(custom.Dynamic_Attack_Refresh_Periods))
    window_id = int((int(time_period) - 1) // refresh_periods)

    if env.dynamic_attack_window_id == window_id:
        return env.current_attack_set

    oracle_num = int(env.oracleNum)
    candidates = list(range(oracle_num))
    if int(getattr(custom, "Dynamic_Exclude_Trusted", 0)) == 1:
        trusted = set(as_int_list(getattr(env.args, "Trusted_Reference_Oracles", [])))
        candidates = [i for i in candidates if i not in trusted]
    if not candidates:
        candidates = list(range(oracle_num))

    if int(custom.Dynamic_Attack_Count) > 0:
        k = int(custom.Dynamic_Attack_Count)
    else:
        k = int(round(float(custom.Dynamic_Attack_Ratio) * oracle_num))
    k = max(1, min(k, len(candidates)))

    active = sorted(env.dynamic_attack_rng.choice(candidates, size=k, replace=False).astype(int).tolist())
    env.current_attack_set = set(active)
    env.ever_attack_set.update(active)
    env.dynamic_attack_window_id = window_id
    env.dynamic_attack_generation += 1
    env.dynamic_attack_window_start_request = int(request_attrs[0])
    env.dynamic_attack_history.append({
        "generation": int(env.dynamic_attack_generation),
        "window_id": int(window_id),
        "period": int(time_period),
        "start_request_id": int(request_attrs[0]) + 1,
        "active_attack_oracles": active,
    })

    apply_dynamic_projection_to_active_set(env, attack_name, custom)
    return env.current_attack_set


def install_attack_runtime_patch(env: SchedulingEnv, attack_name: str, custom):
    attack_name = attack_name.upper()
    ensure_dynamic_attack_state(env, custom)
    original_sim = env._simulate_oracle_attempt

    def update_duration(attempt: Dict, multiplier: float):
        old = float(attempt["durationT"])
        new = max(old * float(multiplier), old)
        delta = new - old
        attempt["durationT"] = new
        attempt["leaveT"] = float(attempt["leaveT"]) + delta
        return attempt

    def patched_sim(request_attrs, action, policy_name, arrival_override=None):
        attempt = original_sim(request_attrs, action, policy_name, arrival_override)
        action = int(action)

        active_set = set(getattr(env, "current_attack_set", set()))
        is_active_attacker = action in active_set

        attempt["attack_mode"] = attack_name
        attempt["dynamic_attack"] = int(getattr(custom, "Dynamic_Attack", 1))
        attempt["attack_generation"] = int(getattr(env, "dynamic_attack_generation", 0))
        attempt["attack_triggered"] = 0
        attempt["attack_phase"] = "clean"
        attempt["is_malicious"] = int(is_active_attacker)

        if not is_active_attacker:
            return attempt

        rid = int(request_attrs[0])
        length = float(request_attrs[2])
        ddl = float(request_attrs[4])

        if attack_name == "ME":
            attempt["attack_phase"] = "dynamic_always_on"
            if np.random.rand() < float(custom.ME_Failure_Prob):
                attempt["validation_raw"] = 0
            if np.random.rand() < float(custom.ME_Severe_Behavior_Prob):
                attempt["behavior_record"] = max(float(attempt["behavior_record"]), 100.0)
            else:
                attempt["behavior_record"] = max(float(attempt["behavior_record"]), 5.0)
            update_duration(attempt, float(custom.ME_Duration_Multiplier))
            attempt["attack_triggered"] = 1

        elif attack_name == "OOA":
            window_len = max(1, int(custom.Dynamic_Attack_Refresh_Periods) * int(getattr(env.args, "Time_Period_Size", 100)))
            local_age = max(0, rid - int(getattr(env, "dynamic_attack_window_start_request", rid)))
            switch_point = int(float(custom.OOA_Window_Warmup_Ratio) * window_len)

            if local_age < switch_point:
                attempt["attack_phase"] = "dynamic_off_clean"
                if np.random.rand() < float(custom.OOA_Off_Force_Clean_Prob):
                    attempt["validation_raw"] = 1
                    attempt["behavior_record"] = 0.0
                return attempt

            attempt["attack_phase"] = "dynamic_on_attack"
            if np.random.rand() < float(custom.OOA_On_Attack_Prob):
                attempt["validation_raw"] = 0
                attempt["behavior_record"] = max(float(attempt["behavior_record"]), 100.0)
                update_duration(attempt, float(custom.OOA_On_Duration_Multiplier))
                attempt["attack_triggered"] = 1

        elif attack_name == "OSA":
            mean_len = float(getattr(env.args, "Request_len_Mean", 6000))
            std_len = max(float(getattr(env.args, "Request_len_Std", 500)), 1.0)
            exe = length / max(float(env.oracleAcc[action]), 1e-8)
            slack = (ddl - exe) / max(ddl, 1e-8)

            high_length = length >= mean_len + float(custom.OSA_Length_Z) * std_len
            tight_deadline = slack <= float(custom.OSA_Slack_Threshold)
            type_match = int(env.oracleTypes[action]) == int(request_attrs[3])

            trigger_score = 0.0
            trigger_score += 0.40 if high_length else 0.0
            trigger_score += 0.40 if tight_deadline else 0.0
            trigger_score += 0.20 if type_match else 0.0
            trigger_prob = float(custom.OSA_Min_Attack_Prob) + trigger_score * float(custom.OSA_Attack_Prob_Gain)
            trigger_prob = float(np.clip(trigger_prob, 0.0, float(custom.OSA_Max_Attack_Prob)))

            attempt["attack_phase"] = "dynamic_opportunistic_candidate" if trigger_score > 0 else "dynamic_stealth_clean"

            if np.random.rand() < trigger_prob:
                attempt["validation_raw"] = 0
                if np.random.rand() < float(custom.OSA_Severe_Behavior_Prob):
                    attempt["behavior_record"] = max(float(attempt["behavior_record"]), 100.0)
                else:
                    attempt["behavior_record"] = max(float(attempt["behavior_record"]), 5.0)
                update_duration(attempt, float(custom.OSA_Duration_Multiplier))
                attempt["attack_triggered"] = 1
                attempt["attack_phase"] = "dynamic_opportunistic_attack"

        if not hasattr(env, "dynamic_attack_attempts"):
            env.dynamic_attack_attempts = []
        env.dynamic_attack_attempts.append({
            "request_id": rid + 1,
            "oracle": action,
            "policy": policy_name,
            "attack": attack_name,
            "generation": int(getattr(env, "dynamic_attack_generation", 0)),
            "attack_triggered": int(attempt.get("attack_triggered", 0)),
            "attack_phase": attempt.get("attack_phase", "clean"),
            "validation_raw": int(attempt.get("validation_raw", 0)),
            "behavior_record": float(attempt.get("behavior_record", 0.0)),
        })
        return attempt

    env._simulate_oracle_attempt = patched_sim


def initialize_attack_priors(env: SchedulingEnv, custom):
    if int(custom.Use_Attack_Prior) == 0:
        return

    attack_set = as_int_list(getattr(env.args, "Attack_Oracles", []))
    trusted_set = as_int_list(getattr(env.args, "Trusted_Reference_Oracles", []))

    for policy in env.policy_names:
        for i in attack_set:
            if 0 <= i < env.oracleNum:
                env.oracle_events[policy][2, i] = min(
                    float(env.oracle_events[policy][2, i]),
                    float(custom.Attack_Prior_Reputation),
                )
                env.audit_beta[policy][i] += float(custom.Attack_Prior_Audit_Beta)

        for i in trusted_set:
            if 0 <= i < env.oracleNum:
                env.audit_alpha[policy][i] += float(custom.Trusted_Prior_Audit_Alpha)


def safe_get_action_mask(env, request_attrs, policy_name: str):
    try:
        return env.get_action_mask(request_attrs, policy_name)
    except TypeError:
        return env.get_action_mask(request_attrs)


def safe_get_backup_action_mask(env, request_attrs, primary, policy_name: str):
    try:
        return env.get_backup_action_mask(request_attrs, primary, policy_name)
    except TypeError:
        return env.get_backup_action_mask(request_attrs, primary)


def is_single_mode(mode_name: str) -> bool:
    return mode_name in ["single_cost", "single_safe"] or mode_name.startswith("single")


def compute_oracle_feature_table(env: SchedulingEnv, policy_name: str, request_attrs, custom, period: int, attack_name: str) -> pd.DataFrame:
    duration, rep, obs, risk, ontime = env._estimated_oracle_metrics(request_attrs, policy_name)
    counts = env.reputation_factors[policy_name][0]
    val = env.reputation_factors[policy_name][1]
    behavior_sum = env.reputation_factors[policy_name][3]

    avg_behavior = behavior_sum / np.maximum(counts, 1.0)
    behavior_risk = np.clip(np.log1p(np.maximum(avg_behavior, 0.0)) / np.log1p(100.0), 0.0, 1.0)
    audit_truth = env.audit_truth_score(policy_name)
    audit_risk = 1.0 - audit_truth

    cost_norm = env.oracleCost / max(float(np.max(env.oracleCost)), 1e-8)
    token_norm = env.oracleToken / max(float(np.max(env.oracleToken)), 1e-8)
    validation_prior = np.asarray(env.oracleValidationProbs, dtype=float)

    active_attack_set = set(getattr(env, "current_attack_set", set(as_int_list(getattr(env.args, "Attack_Oracles", [])))))
    ever_attack_set = set(getattr(env, "ever_attack_set", set(active_attack_set)))
    trusted_set = set(as_int_list(getattr(env.args, "Trusted_Reference_Oracles", [])))

    detected = (
        (risk >= float(custom.Detection_Risk_Threshold))
        | (audit_risk >= float(custom.Detection_Audit_Risk_Threshold))
        | (rep <= float(custom.Detection_Rep_Low_Threshold))
        | (obs <= float(custom.Detection_Obs_Success_Low_Threshold))
    ).astype(int)

    rows = []
    for i in range(env.oracleNum):
        rows.append({
            "attack": attack_name,
            "period": int(period),
            "attack_generation": int(getattr(env, "dynamic_attack_generation", 0)),
            "active_attack_oracles": json.dumps(sorted(list(active_attack_set))),
            "oracle": int(i),
            "is_attack_oracle": int(i in active_attack_set),
            "ever_attack_oracle": int(i in ever_attack_set),
            "is_trusted_reference": int(i in trusted_set),
            "service_type": int(env.oracleTypes[i]),
            "cost_norm": float(cost_norm[i]),
            "token_norm": float(token_norm[i]),
            "validation_prior": float(validation_prior[i]),
            "counts": float(counts[i]),
            "successful_validation_count": float(val[i]),
            "observed_success": float(obs[i]),
            "avg_behavior": float(avg_behavior[i]),
            "behavior_risk": float(behavior_risk[i]),
            "audit_truth": float(audit_truth[i]),
            "audit_risk": float(audit_risk[i]),
            "effective_reputation": float(rep[i]),
            "estimated_duration": float(duration[i]),
            "ontime_prob": float(ontime[i]),
            "hcrl_risk": float(risk[i]),
            "detected_as_malicious": int(detected[i]),
        })
    return pd.DataFrame(rows)


def binary_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)

    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    for k, c in enumerate(counts):
        if c > 1:
            idx = np.where(inverse == k)[0]
            ranks[idx] = float(np.mean(ranks[idx]))

    rank_sum_pos = float(np.sum(ranks[y == 1]))
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def detection_metrics(feature_df: pd.DataFrame) -> Dict[str, float]:
    """Detection under dynamic random attackers is computed over all period-oracle snapshots."""
    if feature_df.empty:
        return {}

    f = feature_df.copy()
    y = f["is_attack_oracle"].astype(int).to_numpy()
    pred = f["detected_as_malicious"].astype(int).to_numpy()

    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    auc = binary_auc(y, f["hcrl_risk"].to_numpy())

    attack_rows = f[f["is_attack_oracle"] == 1]
    benign_rows = f[f["is_attack_oracle"] == 0]

    last_period = int(f["period"].max())
    last_f = f[f["period"] == last_period]
    last_y = last_f["is_attack_oracle"].astype(int).to_numpy()
    last_pred = last_f["detected_as_malicious"].astype(int).to_numpy()
    last_tp = int(np.sum((last_y == 1) & (last_pred == 1)))
    last_fn = int(np.sum((last_y == 1) & (last_pred == 0)))
    last_recall = last_tp / max(last_tp + last_fn, 1)

    return {
        "final_period": last_period,
        "snapshot_count": int(len(f)),
        "positive_attack_snapshots": int(np.sum(y == 1)),
        "malicious_precision": float(precision),
        "malicious_recall": float(recall),
        "malicious_f1": float(f1),
        "benign_specificity": float(specificity),
        "risk_auc": float(auc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "final_period_malicious_recall": float(last_recall),
        "attack_mean_risk": float(attack_rows["hcrl_risk"].mean()) if len(attack_rows) else float("nan"),
        "benign_mean_risk": float(benign_rows["hcrl_risk"].mean()) if len(benign_rows) else float("nan"),
        "attack_mean_rep": float(attack_rows["effective_reputation"].mean()) if len(attack_rows) else float("nan"),
        "benign_mean_rep": float(benign_rows["effective_reputation"].mean()) if len(benign_rows) else float("nan"),
        "attack_mean_audit_risk": float(attack_rows["audit_risk"].mean()) if len(attack_rows) else float("nan"),
        "benign_mean_audit_risk": float(benign_rows["audit_risk"].mean()) if len(benign_rows) else float("nan"),
    }


def run_one_attack(base_args, attack_name: str, models: Dict[str, OptionActorCritic], max_requests: int, custom):
    args, feature_info = apply_attack_feature_engineering(base_args, attack_name, custom)
    env = SchedulingEnv(args)
    install_attack_runtime_patch(env, attack_name, custom)

    env.reset_reputation_factors()
    env.initial_reputation()
    initialize_attack_priors(env, custom)

    policy = POLICY_NAME
    trusted_set = set(as_int_list(getattr(args, "Trusted_Reference_Oracles", [])))
    mode_names = list(getattr(args, "HCRL_Mode_Names", DEFAULT_MODES))

    max_requests = min(int(max_requests), int(args.Request_Num))

    request_logs = []
    period_rows = []
    rep_rows = []
    feature_rows = []

    request_c = 1
    time_period = 1
    last_request_attrs = None

    while request_c <= max_requests:
        finish, request_attrs = env.workload(request_c)
        last_request_attrs = request_attrs
        rid = int(request_attrs[0])

        if request_c > 1 and request_c % int(args.Time_Period_Size) == 0:
            env.update_reputation(env.get_reputation_factors(policy), time_period, policy)

            rep = env._effective_reputation_vector(policy)
            rep_row = {"attack": attack_name, "period": time_period}
            for i, val_i in enumerate(rep):
                rep_row[f"oracle_{i}"] = float(val_i)
            rep_rows.append(rep_row)

            f = compute_oracle_feature_table(env, policy, request_attrs, custom, time_period, attack_name)
            feature_rows.append(f)

            if request_logs:
                df_tmp = pd.DataFrame(request_logs)
                g = df_tmp[df_tmp["period"] == time_period]
                if len(g):
                    period_rows.append({
                        "attack": attack_name,
                        "period": int(time_period),
                        "attack_generation": int(g["attack_generation"].max()),
                        "active_attack_oracles": g["active_attack_oracles"].iloc[-1],
                        "n_requests": int(len(g)),
                        "success_rate": float(g["final_success"].mean()),
                        "primary_success_rate": float(g["primary_success"].mean()),
                        "backup_used_rate": float(g["backup_used"].mean()),
                        "backup_recovery_rate": float(g["backup_recovery"].mean()),
                        "attack_primary_rate": float(g["primary_attacked"].mean()),
                        "attack_backup_rate": float(g["backup_attacked"].mean()),
                        "attack_any_rate": float(g["any_attacked"].mean()),
                        "trusted_primary_rate": float(g["primary_trusted_reference"].mean()),
                        "trusted_backup_rate": float(g["backup_trusted_reference"].mean()),
                        "trusted_any_rate": float(g["any_trusted_reference"].mean()),
                        "safe_primary_rate": float(1.0 - g["primary_attacked"].mean()),
                        "mean_mode_reward": float(g["mode_reward"].mean()),
                        "mean_primary_reward": float(g["primary_reward"].mean()),
                        "mean_backup_reward": float(g["backup_reward"].mean()),
                    })

            env.reset_reputation_factors()
            time_period += 1

        active_set = set(update_dynamic_attack_set(env, request_attrs, time_period, custom, attack_name))

        s_primary = env.getState(request_attrs, policy)
        primary_mask = safe_get_action_mask(env, request_attrs, policy)
        primary = int(models["primary"].choose_best_action(s_primary, primary_mask))

        s_mode = env.get_hcrl_mode_state(request_attrs, policy)
        mode_mask = env.get_hcrl_mode_mask(request_attrs, policy, primary)
        mode_idx = int(models["mode"].choose_best_action(s_mode, mode_mask))
        mode_name = mode_names[mode_idx]

        if is_single_mode(mode_name):
            backup = -1
        else:
            backup_mask = safe_get_backup_action_mask(env, request_attrs, primary, policy)
            backup = int(models["backup"].choose_best_action(s_primary, backup_mask))

        counts_before = env.reputation_factors[policy][0].copy()
        val_before = env.reputation_factors[policy][1].copy()
        beh_before = env.reputation_factors[policy][3].copy()

        feedback = env.feedback_hcrl(request_attrs, mode_idx, primary, backup, policy)

        counts_after = env.reputation_factors[policy][0].copy()
        val_after = env.reputation_factors[policy][1].copy()
        beh_after = env.reputation_factors[policy][3].copy()

        primary_count_delta = float(counts_after[primary] - counts_before[primary])
        primary_val_delta = float(val_after[primary] - val_before[primary])
        primary_beh_delta = float(beh_after[primary] - beh_before[primary])

        backup_count_delta = 0.0
        backup_val_delta = 0.0
        backup_beh_delta = 0.0
        if backup is not None and int(backup) >= 0 and int(backup) < env.oracleNum:
            backup_count_delta = float(counts_after[backup] - counts_before[backup])
            backup_val_delta = float(val_after[backup] - val_before[backup])
            backup_beh_delta = float(beh_after[backup] - beh_before[backup])

        pb = env.pb_records[policy]
        final_success = int(pb[0, rid] or pb[2, rid] or env.events[policy][7, rid])
        primary_success = int(pb[0, rid])
        backup_used = int(pb[1, rid])
        backup_recovery = int(pb[3, rid])
        recorded_backup = int(pb[5, rid]) if pb[5, rid] >= 0 else -1

        request_logs.append({
            "attack": attack_name,
            "request_id": int(rid + 1),
            "period": int(time_period),
            "attack_generation": int(getattr(env, "dynamic_attack_generation", 0)),
            "active_attack_oracles": json.dumps(sorted(list(active_set))),
            "request_type": int(request_attrs[3]),
            "request_length": float(request_attrs[2]),
            "deadline": float(request_attrs[4]),
            "mode_index": int(mode_idx),
            "mode_name": mode_name,
            "primary_oracle": int(primary),
            "backup_oracle": int(recorded_backup),
            "primary_attacked": int(primary in active_set),
            "backup_attacked": int(recorded_backup in active_set),
            "any_attacked": int(primary in active_set or recorded_backup in active_set),
            "primary_trusted_reference": int(primary in trusted_set),
            "backup_trusted_reference": int(recorded_backup in trusted_set),
            "any_trusted_reference": int(primary in trusted_set or recorded_backup in trusted_set),
            "primary_success": primary_success,
            "backup_used": backup_used,
            "backup_recovery": backup_recovery,
            "final_success": final_success,
            "primary_observed_count_delta": primary_count_delta,
            "primary_observed_validation_delta": primary_val_delta,
            "primary_observed_behavior_delta": primary_beh_delta,
            "backup_observed_count_delta": backup_count_delta,
            "backup_observed_validation_delta": backup_val_delta,
            "backup_observed_behavior_delta": backup_beh_delta,
            "mode_reward": float(feedback.get("mode_reward", 0.0)),
            "primary_reward": float(feedback.get("primary_reward", 0.0)),
            "backup_reward": float(feedback.get("backup_reward", 0.0)),
        })

        request_c += 1
        if finish:
            break

    if last_request_attrs is not None:
        try:
            env.update_reputation(env.get_reputation_factors(policy), time_period, policy)
            rep = env._effective_reputation_vector(policy)
            rep_row = {"attack": attack_name, "period": time_period}
            for i, val_i in enumerate(rep):
                rep_row[f"oracle_{i}"] = float(val_i)
            rep_rows.append(rep_row)
            feature_rows.append(compute_oracle_feature_table(env, policy, last_request_attrs, custom, time_period, attack_name))
        except Exception as e:
            print(f"[warn] final reputation snapshot failed for {attack_name}: {e}")

    req_df = pd.DataFrame(request_logs)
    period_df = pd.DataFrame(period_rows)
    rep_df = pd.DataFrame(rep_rows)
    feature_df = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()

    det = detection_metrics(feature_df)
    dynamic_history = getattr(env, "dynamic_attack_history", [])
    overall = {
        "attack": attack_name,
        "n_requests": int(len(req_df)),
        "dynamic_attack": int(getattr(custom, "Dynamic_Attack", 1)),
        "dynamic_generations": int(len(dynamic_history)),
        "dynamic_attack_ratio": float(getattr(custom, "Dynamic_Attack_Ratio", 0.0)),
        "dynamic_attack_count": int(getattr(custom, "Dynamic_Attack_Count", 0)),
        "active_attack_oracles_last": json.dumps(sorted(list(getattr(env, "current_attack_set", set())))),
        "ever_attack_oracles": json.dumps(sorted(list(getattr(env, "ever_attack_set", set())))),
        "trusted_reference_oracles": json.dumps(sorted(list(trusted_set))),
        "attack_primary_rate": float(req_df["primary_attacked"].mean()) if len(req_df) else 0.0,
        "attack_backup_rate": float(req_df["backup_attacked"].mean()) if len(req_df) else 0.0,
        "attack_any_rate": float(req_df["any_attacked"].mean()) if len(req_df) else 0.0,
        "trusted_primary_rate": float(req_df["primary_trusted_reference"].mean()) if len(req_df) else 0.0,
        "trusted_backup_rate": float(req_df["backup_trusted_reference"].mean()) if len(req_df) else 0.0,
        "trusted_any_rate": float(req_df["any_trusted_reference"].mean()) if len(req_df) else 0.0,
        "safe_primary_rate": float(1.0 - req_df["primary_attacked"].mean()) if len(req_df) else 0.0,
        "success_rate": float(req_df["final_success"].mean()) if len(req_df) else 0.0,
        "backup_used_rate": float(req_df["backup_used"].mean()) if len(req_df) else 0.0,
        "backup_recovery_rate": float(req_df["backup_recovery"].mean()) if len(req_df) else 0.0,
        **det,
    }

    history_df = pd.DataFrame(dynamic_history)
    feature_info["dynamic_history_rows"] = int(len(history_df))

    return req_df, period_df, rep_df, feature_df, overall, feature_info


def plot_outputs(period_df: pd.DataFrame, rep_df: pd.DataFrame, feature_df: pd.DataFrame, out_dir: Path, fmt: str = "png"):
    if plt is None:
        print("[warn] matplotlib unavailable; skip plots")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if period_df is not None and len(period_df):
        for attack in sorted(period_df["attack"].unique()):
            p = period_df[period_df["attack"] == attack]
            if not len(p):
                continue

            plt.figure(figsize=(10, 4))
            plt.plot(p["period"], p["attack_any_rate"], marker="o", label="malicious selected (any)")
            plt.plot(p["period"], p["attack_primary_rate"], marker="o", label="malicious selected as primary")
            plt.plot(p["period"], p["trusted_any_rate"], marker="o", label="trusted reference selected (any)")
            plt.plot(p["period"], p["safe_primary_rate"], marker="o", label="safe primary rate")
            if "success_rate" in p.columns:
                plt.plot(p["period"], p["success_rate"], marker="o", label="success rate")
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Time period")
            plt.ylabel("Empirical probability")
            plt.title(f"{attack}: HCRL selection / avoidance")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"{attack}_selection_probability.{fmt}", dpi=300)
            plt.close()

    if feature_df is not None and len(feature_df):
        for attack in sorted(feature_df["attack"].unique()):
            f = feature_df[feature_df["attack"] == attack]

            grouped = f.groupby(["period", "is_attack_oracle"]).agg(
                mean_risk=("hcrl_risk", "mean"),
                mean_rep=("effective_reputation", "mean"),
                mean_audit_risk=("audit_risk", "mean"),
                detected_rate=("detected_as_malicious", "mean"),
            ).reset_index()

            plt.figure(figsize=(10, 4))
            for is_mal, label in [(1, "malicious"), (0, "benign")]:
                g = grouped[grouped["is_attack_oracle"] == is_mal]
                if len(g):
                    plt.plot(g["period"], g["mean_risk"], marker="o", label=f"{label} mean HCRL risk")
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Time period")
            plt.ylabel("Risk")
            plt.title(f"{attack}: engineered malicious risk separation")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"{attack}_risk_separation.{fmt}", dpi=300)
            plt.close()

            plt.figure(figsize=(10, 4))
            for is_mal, label in [(1, "malicious"), (0, "benign")]:
                g = grouped[grouped["is_attack_oracle"] == is_mal]
                if len(g):
                    plt.plot(g["period"], g["mean_rep"], marker="o", label=f"{label} mean reputation")
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Time period")
            plt.ylabel("Effective reputation")
            plt.title(f"{attack}: reputation trajectory")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"{attack}_reputation_grouped.{fmt}", dpi=300)
            plt.close()

            plt.figure(figsize=(10, 4))
            for is_mal, label in [(1, "malicious"), (0, "benign")]:
                g = grouped[grouped["is_attack_oracle"] == is_mal]
                if len(g):
                    plt.plot(g["period"], g["detected_rate"], marker="o", label=f"{label} detected rate")
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Time period")
            plt.ylabel("Detection rate")
            plt.title(f"{attack}: malicious identification by HCRL risk threshold")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"{attack}_detection_rate.{fmt}", dpi=300)
            plt.close()

    if rep_df is not None and len(rep_df):
        oracle_cols = [c for c in rep_df.columns if c.startswith("oracle_")]
        for attack in sorted(rep_df["attack"].unique()):
            r = rep_df[rep_df["attack"] == attack]
            if not len(r) or not oracle_cols:
                continue

            f_attack = feature_df[(feature_df["attack"] == attack) & (feature_df["is_attack_oracle"] == 1)]
            attack_oracles = sorted(f_attack["oracle"].unique().tolist()) if len(f_attack) else []
            candidate_cols = [f"oracle_{i}" for i in attack_oracles if f"oracle_{i}" in oracle_cols]
            candidate_cols += [c for c in oracle_cols if c not in candidate_cols][:4]

            plt.figure(figsize=(10, 4))
            for c in candidate_cols:
                plt.plot(r["period"], r[c], marker="o", label=c)
            plt.xlabel("Time period")
            plt.ylabel("Effective reputation")
            plt.title(f"{attack}: selected oracle reputation trajectories")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(fig_dir / f"{attack}_reputation_selected_oracles.{fmt}", dpi=300)
            plt.close()


def parse_custom_args():
    custom = argparse.ArgumentParser(add_help=False)

    custom.add_argument("--Weight_Dir", type=str, default="auto",
                        help="Default auto-resolves to TCO-DRL/weight/audit_hcrl.")
    custom.add_argument("--Weight_Service_Type_Num", type=int, default=0,
                        help="Optional: service type count used during HCRL training. 0 = infer.")
    custom.add_argument("--Weight_Oracles_Per_Type", type=int, default=0,
                        help="Optional: oracles per service type used during HCRL training. 0 = infer.")
    custom.add_argument("--Attacks", nargs="+", default=["ME", "OOA", "OSA"],
                        help="Attack profiles to evaluate: ME OOA OSA.")
    custom.add_argument("--Attack_Oracles", nargs="*", type=int, default=None,
                        help="Static ablation only. In Experiment C dynamic mode, this is ignored.")
    custom.add_argument("--Dynamic_Attack", type=int, default=1,
                        help="1 = Experiment C dynamic random attackers; 0 = static attacker ablation.")
    custom.add_argument("--Dynamic_Attack_Ratio", type=float, default=0.20,
                        help="Fraction of oracle community randomly activated as attackers per window.")
    custom.add_argument("--Dynamic_Attack_Count", type=int, default=0,
                        help="Number of active attackers per window. 0 = use Dynamic_Attack_Ratio.")
    custom.add_argument("--Dynamic_Attack_Refresh_Periods", type=int, default=1,
                        help="Resample active attackers every N reputation periods.")
    custom.add_argument("--Dynamic_Attack_Seed", type=int, default=-1,
                        help="Random seed for dynamic attacker sampling. -1 = args.Seed + 7919.")
    custom.add_argument("--Dynamic_Exclude_Trusted", type=int, default=0,
                        help="1 = do not sample trusted reference oracles as attackers; 0 = random from all oracles.")
    custom.add_argument("--Dynamic_Projection", type=int, default=1,
                        help="1 = project active random attackers to the training-time malicious feature prototype.")
    custom.add_argument("--Max_Requests", type=int, default=3000)
    custom.add_argument("--Attack_Output_Dir", type=str, default="attack_weight_eval_output")
    custom.add_argument("--Figure_Format", type=str, default="png", choices=["png", "jpg", "svg", "pdf"])

    custom.add_argument("--Detection_Risk_Threshold", type=float, default=0.55)
    custom.add_argument("--Detection_Audit_Risk_Threshold", type=float, default=0.55)
    custom.add_argument("--Detection_Rep_Low_Threshold", type=float, default=0.35)
    custom.add_argument("--Detection_Obs_Success_Low_Threshold", type=float, default=0.45)

    custom.add_argument("--Use_Attack_Prior", type=int, default=0)
    custom.add_argument("--Attack_Prior_Reputation", type=float, default=0.35)
    custom.add_argument("--Attack_Prior_Audit_Beta", type=float, default=1.50)
    custom.add_argument("--Trusted_Prior_Audit_Alpha", type=float, default=0.50)

    custom.add_argument("--ME_Cost_Scale", type=float, default=0.55)
    custom.add_argument("--ME_Acc_Scale", type=float, default=0.90)
    custom.add_argument("--ME_Token_Scale", type=float, default=0.25)
    custom.add_argument("--ME_Validation_Prob", type=float, default=0.05)
    custom.add_argument("--ME_Behavior_P0", type=float, default=0.03)
    custom.add_argument("--ME_Behavior_P1", type=float, default=0.07)
    custom.add_argument("--ME_Behavior_P5", type=float, default=0.25)
    custom.add_argument("--ME_Behavior_P100", type=float, default=0.65)
    custom.add_argument("--ME_Fatigue", type=float, default=0.45)
    custom.add_argument("--ME_Failure_Prob", type=float, default=0.90)
    custom.add_argument("--ME_Severe_Behavior_Prob", type=float, default=0.75)
    custom.add_argument("--ME_Duration_Multiplier", type=float, default=1.25)

    custom.add_argument("--OOA_Cost_Scale", type=float, default=0.60)
    custom.add_argument("--OOA_Acc_Scale", type=float, default=1.00)
    custom.add_argument("--OOA_Token_Scale", type=float, default=0.35)
    custom.add_argument("--OOA_Off_Validation_Prob", type=float, default=0.88)
    custom.add_argument("--OOA_Off_Behavior_P0", type=float, default=0.85)
    custom.add_argument("--OOA_Off_Behavior_P1", type=float, default=0.12)
    custom.add_argument("--OOA_Off_Behavior_P5", type=float, default=0.03)
    custom.add_argument("--OOA_Off_Behavior_P100", type=float, default=0.00)
    custom.add_argument("--OOA_Fatigue", type=float, default=0.20)
    custom.add_argument("--OOA_On_Ratio", type=float, default=0.35,
                        help="Static fallback only. After this fraction of requests, OOA switches to attack phase.")
    custom.add_argument("--OOA_Window_Warmup_Ratio", type=float, default=0.35,
                        help="Dynamic OOA: clean warm-up fraction inside each random attack window.")
    custom.add_argument("--OOA_Off_Force_Clean_Prob", type=float, default=0.90)
    custom.add_argument("--OOA_On_Attack_Prob", type=float, default=0.85)
    custom.add_argument("--OOA_On_Duration_Multiplier", type=float, default=1.18)

    custom.add_argument("--OSA_Cost_Scale", type=float, default=0.65)
    custom.add_argument("--OSA_Acc_Scale", type=float, default=0.96)
    custom.add_argument("--OSA_Token_Scale", type=float, default=0.40)
    custom.add_argument("--OSA_Base_Validation_Prob", type=float, default=0.72)
    custom.add_argument("--OSA_Base_Behavior_P0", type=float, default=0.68)
    custom.add_argument("--OSA_Base_Behavior_P1", type=float, default=0.20)
    custom.add_argument("--OSA_Base_Behavior_P5", type=float, default=0.10)
    custom.add_argument("--OSA_Base_Behavior_P100", type=float, default=0.02)
    custom.add_argument("--OSA_Fatigue", type=float, default=0.25)
    custom.add_argument("--OSA_Length_Z", type=float, default=0.50)
    custom.add_argument("--OSA_Slack_Threshold", type=float, default=0.20)
    custom.add_argument("--OSA_Min_Attack_Prob", type=float, default=0.05)
    custom.add_argument("--OSA_Attack_Prob_Gain", type=float, default=0.80)
    custom.add_argument("--OSA_Max_Attack_Prob", type=float, default=0.90)
    custom.add_argument("--OSA_Severe_Behavior_Prob", type=float, default=0.50)
    custom.add_argument("--OSA_Duration_Multiplier", type=float, default=1.15)

    custom_args, rest = custom.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    base_args = get_args()
    configure_hcrl_eval_args(base_args)
    return custom_args, base_args


def main():
    custom, args = parse_custom_args()

    np.random.seed(int(args.Seed))

    weight_dir = resolve_weight_dir(custom.Weight_Dir)
    if not weight_dir.exists():
        raise FileNotFoundError(
            f"Weight_Dir does not exist: {weight_dir}\n"
            f"Expected your weights under TCO-DRL/weight/audit_hcrl. "
            f"Pass --Weight_Dir /path/to/TCO-DRL/weight/audit_hcrl if needed."
        )

    weight_shapes = inspect_hcrl_weight_shapes(weight_dir)
    print("[weights] checkpoint shapes:")
    for role, info in weight_shapes.items():
        print(
            f"  - {role}: input_dim={info['input_dim']}, "
            f"hidden={info['hidden_dim']}, actions={info.get('n_actions')}, "
            f"path={info['path']}"
        )

    args = align_args_to_hcrl_weights(args, weight_shapes, custom)

    env0 = SchedulingEnv(args)
    models = build_hcrl_models(args, env0, weight_dir)

    run_id = datetime.now().strftime("hcrl_attack_feature_eval_%Y%m%d_%H%M%S")
    out_dir = Path(custom.Attack_Output_Dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    request_logs = []
    period_logs = []
    rep_logs = []
    feature_logs = []
    overall_rows = []
    feature_info_rows = []

    for attack in custom.Attacks:
        attack = attack.upper().strip()
        print(f"[attack] running {attack}")

        req_df, period_df, rep_df, feature_df, overall, feature_info = run_one_attack(
            args, attack, models, custom.Max_Requests, custom
        )

        request_logs.append(req_df)
        period_logs.append(period_df)
        rep_logs.append(rep_df)
        feature_logs.append(feature_df)
        overall_rows.append(overall)
        feature_info_rows.append(feature_info)

        print(
            f"[attack] {attack}: "
            f"safe_primary={overall.get('safe_primary_rate', 0.0):.3f}, "
            f"malicious_recall={overall.get('malicious_recall', float('nan')):.3f}, "
            f"malicious_f1={overall.get('malicious_f1', float('nan')):.3f}, "
            f"risk_auc={overall.get('risk_auc', float('nan')):.3f}"
        )

    req_all = pd.concat(request_logs, ignore_index=True) if request_logs else pd.DataFrame()
    period_all = pd.concat(period_logs, ignore_index=True) if period_logs else pd.DataFrame()
    rep_all = pd.concat(rep_logs, ignore_index=True) if rep_logs else pd.DataFrame()
    feature_all = pd.concat(feature_logs, ignore_index=True) if feature_logs else pd.DataFrame()
    overall_all = pd.DataFrame(overall_rows)
    feature_info_all = pd.DataFrame(feature_info_rows)

    req_all.to_csv(out_dir / "attack_request_log.csv", index=False, encoding="utf-8-sig")
    period_all.to_csv(out_dir / "attack_period_summary.csv", index=False, encoding="utf-8-sig")
    rep_all.to_csv(out_dir / "attack_reputation_trajectory.csv", index=False, encoding="utf-8-sig")
    feature_all.to_csv(out_dir / "attack_oracle_feature_log.csv", index=False, encoding="utf-8-sig")
    overall_all.to_csv(out_dir / "attack_overall_summary.csv", index=False, encoding="utf-8-sig")
    overall_all.to_csv(out_dir / "attack_detection_summary.csv", index=False, encoding="utf-8-sig")
    feature_info_all.to_csv(out_dir / "attack_feature_engineering_config.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "custom_args": vars(custom),
                "base_args": vars(args),
                "weight_dir": str(weight_dir),
                "output_dir": str(out_dir.resolve()),
                "feature_engineering": feature_info_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    plot_outputs(period_all, rep_all, feature_all, out_dir, custom.Figure_Format)

    print("\nSaved outputs to:", out_dir.resolve())
    if len(overall_all):
        display_cols = [
            "attack", "n_requests", "safe_primary_rate", "attack_any_rate",
            "success_rate", "malicious_precision", "malicious_recall",
            "malicious_f1", "risk_auc", "attack_mean_risk", "benign_mean_risk",
            "attack_mean_rep", "benign_mean_rep",
        ]
        display_cols = [c for c in display_cols if c in overall_all.columns]
        print(overall_all[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
