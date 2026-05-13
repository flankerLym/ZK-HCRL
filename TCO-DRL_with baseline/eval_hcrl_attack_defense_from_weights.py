"""
Evaluate saved HCRL .npz weights under ME / OOA / OSA attacks in the baseline simulator.

Place this file in:
    TCO-DRL_with baseline/eval_hcrl_attack_defense_from_weights.py

Run after applying apply_hcrl_me_ooa_osa_patch.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from utils import get_args
from env import SchedulingEnv
from model import OptionActorCritic


def load_npz_checkpoint(model, path: Path, strict=True):
    if not path.exists():
        raise FileNotFoundError(path)
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
        raise ValueError(f"No compatible arrays in {path}; keys={data.files}")


def build_hcrl_models(args, env, weight_dir: Path):
    mode_dim = env.s_features + env.mode_extra_features
    models = {
        "mode": OptionActorCritic(
            len(args.HCRL_Mode_Names), mode_dim, hidden_units=max(64, args.Dqn_hidden // 2),
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
    load_npz_checkpoint(models["mode"], weight_dir / "HCRL_Mode.npz")
    load_npz_checkpoint(models["primary"], weight_dir / "HCRL_Primary.npz")
    load_npz_checkpoint(models["backup"], weight_dir / "HCRL_Backup.npz")
    return models


def safe_bool_index(values, idx_set):
    return [int(v) in idx_set for v in values]


def run_one_attack(base_args, attack_name: str, models, max_requests: int):
    args = base_args
    args.Attack_Mode = attack_name
    if "HCRL-Oracle" not in args.Baselines:
        args.Baselines = ["HCRL-Oracle"]
        args.Baseline_num = 1
    env = SchedulingEnv(args)
    env.reset_reputation_factors()
    env.initial_reputation()

    attack_set = set(int(x) for x in getattr(args, "Attack_Oracles", []))
    trusted_set = set(int(x) for x in getattr(args, "Trusted_Reference_Oracles", []))
    policy = "HCRL-Oracle"
    logs = []
    period_rows = []
    rep_rows = []

    request_c = 1
    time_period = 1
    max_requests = min(int(max_requests), int(args.Request_Num))
    while request_c <= max_requests:
        if request_c % args.Time_Period_Size == 0:
            # update reputation and save period snapshot before clearing factors
            env.update_reputation(env.get_reputation_factors(policy), time_period, policy)
            rep = env._effective_reputation_vector(policy)
            rep_row = {"attack": attack_name, "period": time_period}
            for i, val in enumerate(rep):
                rep_row[f"oracle_{i}"] = float(val)
            rep_rows.append(rep_row)
            env.reset_reputation_factors()
            time_period += 1

        finish, request_attrs = env.workload(request_c)
        s_primary = env.getState(request_attrs, policy)
        primary_mask = env.get_action_mask(request_attrs, policy)
        primary = int(models["primary"].choose_best_action(s_primary, primary_mask))

        s_mode = env.get_hcrl_mode_state(request_attrs, policy)
        mode_mask = env.get_hcrl_mode_mask(request_attrs, policy, primary)
        mode_idx = int(models["mode"].choose_best_action(s_mode, mode_mask))
        mode_name = args.HCRL_Mode_Names[mode_idx]

        if mode_name in ["single_cost", "single_safe"] or mode_name.startswith("single"):
            backup = -1
        else:
            backup_mask = env.get_backup_action_mask(request_attrs, primary, policy)
            backup = int(models["backup"].choose_best_action(s_primary, backup_mask))

        feedback = env.feedback_hcrl(request_attrs, mode_idx, primary, backup, policy)
        logs.append({
            "attack": attack_name,
            "request_id": int(request_attrs[0]) + 1,
            "period": int(time_period),
            "request_type": int(request_attrs[3]),
            "mode_index": int(mode_idx),
            "mode_name": mode_name,
            "primary_oracle": int(primary),
            "backup_oracle": int(backup),
            "primary_attacked": int(primary in attack_set),
            "backup_attacked": int(backup in attack_set),
            "any_attacked": int(primary in attack_set or backup in attack_set),
            "primary_trusted_reference": int(primary in trusted_set),
            "backup_trusted_reference": int(backup in trusted_set),
            "any_trusted_reference": int(primary in trusted_set or backup in trusted_set),
            "primary_reward": float(feedback.get("primary_reward", 0.0)),
            "backup_reward": float(feedback.get("backup_reward", 0.0)),
            "mode_reward": float(feedback.get("mode_reward", 0.0)),
        })
        request_c += 1
        if finish:
            break

    # final reputation snapshot
    try:
        env.update_reputation(env.get_reputation_factors(policy), time_period, policy)
        rep = env._effective_reputation_vector(policy)
        rep_row = {"attack": attack_name, "period": time_period}
        for i, val in enumerate(rep):
            rep_row[f"oracle_{i}"] = float(val)
        rep_rows.append(rep_row)
    except Exception:
        pass

    df = pd.DataFrame(logs)
    if len(df):
        for period, g in df.groupby("period"):
            rep_vec = env.oracle_reputation_history.get(policy, np.zeros((1, env.oracleNum)))
            row = {
                "attack": attack_name,
                "period": int(period),
                "n_requests": int(len(g)),
                "attacked_primary_rate": float(g["primary_attacked"].mean()),
                "attacked_backup_rate": float(g["backup_attacked"].mean()),
                "attacked_any_rate": float(g["any_attacked"].mean()),
                "trusted_primary_rate": float(g["primary_trusted_reference"].mean()),
                "trusted_backup_rate": float(g["backup_trusted_reference"].mean()),
                "trusted_any_rate": float(g["any_trusted_reference"].mean()),
                "safe_primary_rate": float(1.0 - g["primary_attacked"].mean()),
            }
            period_rows.append(row)

    overall = {
        "attack": attack_name,
        "n_requests": int(len(df)),
        "attacked_primary_rate": float(df["primary_attacked"].mean()) if len(df) else 0.0,
        "attacked_backup_rate": float(df["backup_attacked"].mean()) if len(df) else 0.0,
        "attacked_any_rate": float(df["any_attacked"].mean()) if len(df) else 0.0,
        "trusted_primary_rate": float(df["primary_trusted_reference"].mean()) if len(df) else 0.0,
        "trusted_backup_rate": float(df["backup_trusted_reference"].mean()) if len(df) else 0.0,
        "trusted_any_rate": float(df["any_trusted_reference"].mean()) if len(df) else 0.0,
        "safe_primary_rate": float(1.0 - df["primary_attacked"].mean()) if len(df) else 0.0,
    }
    return df, pd.DataFrame(period_rows), pd.DataFrame(rep_rows), overall


def plot_outputs(period_df, rep_df, out_dir: Path, fmt="png"):
    if plt is None:
        print("[warn] matplotlib unavailable; skip plots")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for attack in sorted(period_df["attack"].unique()):
        p = period_df[period_df["attack"] == attack]
        if len(p):
            plt.figure(figsize=(10, 4))
            plt.plot(p["period"], p["attacked_any_rate"], marker="o", label="attacked selected (any)")
            plt.plot(p["period"], p["attacked_primary_rate"], marker="o", label="attacked selected as primary")
            plt.plot(p["period"], p["trusted_any_rate"], marker="o", label="trusted reference selected (any)")
            plt.plot(p["period"], p["safe_primary_rate"], marker="o", label="safe primary rate")
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Time period")
            plt.ylabel("Empirical probability")
            plt.title(f"{attack}: selection probability by period")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"{attack}_selection_probability.{fmt}", dpi=300)
            plt.close()
        r = rep_df[rep_df["attack"] == attack]
        oracle_cols = [c for c in r.columns if c.startswith("oracle_")]
        if len(r) and oracle_cols:
            plt.figure(figsize=(10, 4))
            for c in oracle_cols:
                idx = int(c.split("_")[1])
                if idx in set([4, 19, 29, 3, 14, 24]):
                    plt.plot(r["period"], r[c], marker="o", label=c)
            plt.xlabel("Time period")
            plt.ylabel("Effective reputation")
            plt.title(f"{attack}: reputation trajectories")
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(fig_dir / f"{attack}_reputation.{fmt}", dpi=300)
            plt.close()


def parse_custom_args():
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--Weight_Dir", type=str, default="../TCO-DRL_on blockchain/weight/audit_hcrl")
    custom.add_argument("--Attacks", nargs="+", default=["ME", "OOA", "OSA"])
    custom.add_argument("--Max_Requests", type=int, default=3000)
    custom.add_argument("--Attack_Output_Dir", type=str, default="attack_weight_eval_output")
    custom.add_argument("--Figure_Format", type=str, default="png", choices=["png", "jpg", "svg", "pdf"])
    custom_args, rest = custom.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    base_args = get_args()
    return custom_args, base_args


def main():
    custom, args = parse_custom_args()
    np.random.seed(args.Seed)
    if "HCRL-Oracle" not in args.Baselines:
        args.Baselines = ["HCRL-Oracle"]
        args.Baseline_num = 1
    weight_dir = Path(custom.Weight_Dir)
    if not weight_dir.is_absolute():
        weight_dir = (Path.cwd() / weight_dir).resolve()

    # Build env once to construct model shapes, then re-create per attack.
    env0 = SchedulingEnv(args)
    models = build_hcrl_models(args, env0, weight_dir)

    run_id = datetime.now().strftime("hcrl_weight_attack_%Y%m%d_%H%M%S")
    out_dir = Path(custom.Attack_Output_Dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    request_logs, period_logs, rep_logs, overall_rows = [], [], [], []
    for attack in custom.Attacks:
        attack = attack.upper()
        print(f"[attack] running {attack}")
        df_req, df_period, df_rep, overall = run_one_attack(args, attack, models, custom.Max_Requests)
        request_logs.append(df_req)
        period_logs.append(df_period)
        rep_logs.append(df_rep)
        overall_rows.append(overall)

    req_all = pd.concat(request_logs, ignore_index=True) if request_logs else pd.DataFrame()
    period_all = pd.concat(period_logs, ignore_index=True) if period_logs else pd.DataFrame()
    rep_all = pd.concat(rep_logs, ignore_index=True) if rep_logs else pd.DataFrame()
    overall_all = pd.DataFrame(overall_rows)

    req_all.to_csv(out_dir / "attack_request_log.csv", index=False, encoding="utf-8-sig")
    period_all.to_csv(out_dir / "attack_period_summary.csv", index=False, encoding="utf-8-sig")
    rep_all.to_csv(out_dir / "attack_reputation_trajectory.csv", index=False, encoding="utf-8-sig")
    overall_all.to_csv(out_dir / "attack_overall_summary.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"custom_args": vars(custom), "args": vars(args), "weight_dir": str(weight_dir)}, f, indent=2, ensure_ascii=False, default=str)
    if len(period_all) and len(rep_all):
        plot_outputs(period_all, rep_all, out_dir, custom.Figure_Format)
    print("Saved outputs to:", out_dir.resolve())
    print(overall_all)


if __name__ == "__main__":
    main()
