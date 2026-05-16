#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_hcrl_convergence_png.py

只画 PNG：不同学习率下 HCRL 的敛散性曲线。
相比旧版：
1. 固定更宽的 y 轴范围，避免纵坐标自动缩放导致抖动被夸大。
2. 默认 smooth=7。
3. 减少 marker 密度，论文图更干净。
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Y_LIMITS = {
    "reward": (0, 2300),
    "success_rate": (50, 85),
    "success_time_rate": (80, 100),
    "malicious_rate": (0, 10),
    "trusted_rate": (70, 100),
    "cost_per_success": (1.0, 2.3),
    "backup_recovery_rate": (0, 45),
    "conditional_backup_recovery_rate": (0, 45),
    "single_mode_rate": (0, 15),
    "serial_mode_rate": (0, 45),
    "parallel_mode_rate": (50, 85),
}


def parse_lr_from_text(text: str, path: Path) -> Optional[float]:
    s = path.parent.name + " " + path.name + " " + text[:5000]

    m = re.search(r"convergence_coupled_lr_([0-9]+p[0-9]+)", s)
    if m:
        return float(m.group(1).replace("p", "."))

    m = re.search(r"hcrl_lr_([0-9]+p[0-9]+)", s)
    if m:
        return float(m.group(1).replace("p", "."))

    m = re.search(r"cv_l([0-9]+p[0-9]+)", s)
    if m:
        raw = float(m.group(1).replace("p", "."))
        return raw / 10000.0

    m = re.search(r"HCRL_lr[=\s]+([0-9]*\.?[0-9]+)", s)
    if m:
        return float(m.group(1))

    return None


def parse_seed(text: str, path: Path) -> Optional[int]:
    s = path.parent.name + " " + path.name + " " + text[:5000]
    for pat in [r"Seed(\d+)", r"seed[_-](\d+)", r"--Seed\s+(\d+)"]:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def parse_log(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lr = parse_lr_from_text(text, path)
    if lr is None:
        return pd.DataFrame()

    seed = parse_seed(text, path)
    episode = None
    rows: List[Dict[str, object]] = []
    pending_main = {}

    ep_re = re.compile(r"Episode\s+(\d+)")
    hcrl_re = re.compile(r"^\[HCRL-Oracle\]\s+(.*)$")
    diag_re = re.compile(r"^\[HCRL-Oracle diagnostics\]\s+(.*)$")
    kv_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(-?\d+(?:\.\d+)?)\s*%?")

    for raw in text.splitlines():
        line = raw.strip()

        m_ep = ep_re.search(line)
        if m_ep:
            episode = int(m_ep.group(1))
            continue

        m_main = hcrl_re.match(line)
        if m_main and "diagnostics" not in line:
            vals = {k: float(v) for k, v in kv_re.findall(m_main.group(1))}
            vals.update({
                "episode": episode,
                "lr": lr,
                "seed": seed,
                "run_id": path.parent.name,
                "log_file": str(path),
            })
            pending_main[episode] = vals
            continue

        m_diag = diag_re.match(line)
        if m_diag:
            vals = {k: float(v) for k, v in kv_re.findall(m_diag.group(1))}
            base = pending_main.get(episode, {
                "episode": episode,
                "lr": lr,
                "seed": seed,
                "run_id": path.parent.name,
                "log_file": str(path),
            })
            base.update(vals)
            rows.append(base)
            pending_main.pop(episode, None)

    rows.extend(pending_main.values())
    return pd.DataFrame(rows)


def discover_logs(input_dir: str, tag_filter: str) -> List[Path]:
    root = Path(input_dir)
    if root.is_file():
        logs = [root]
    else:
        logs = list(root.rglob("*.txt"))

    if tag_filter:
        logs = [p for p in logs if tag_filter.lower() in str(p).lower()]
    return logs


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path, smooth: int = 7):
    if metric not in df.columns:
        print(f"[Skip] metric not found: {metric}")
        return

    data = df.dropna(subset=["episode", "lr", metric]).copy()
    if data.empty:
        print(f"[Skip] empty metric: {metric}")
        return

    plt.figure(figsize=(9, 5))
    for lr in sorted(data["lr"].unique()):
        g = data[data["lr"] == lr]
        curve = g.groupby("episode")[metric].agg(["mean", "std"]).sort_index()

        x = curve.index.values
        y = moving_average(curve["mean"].values, smooth)

        plt.plot(
            x,
            y,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            markevery=5,
            label=f"lr={lr:g}",
        )

        if curve["std"].notna().any() and g["seed"].nunique() > 1:
            std = curve["std"].fillna(0.0).values
            plt.fill_between(x, y - std, y + std, alpha=0.10)

    if metric in Y_LIMITS:
        plt.ylim(*Y_LIMITS[metric])

    plt.xlabel("Episode")
    ylabel = metric
    if metric.endswith("_rate") or metric in [
        "success_rate", "success_time_rate", "malicious_rate", "trusted_rate",
        "backup_used_rate", "backup_recovery_rate", "conditional_backup_recovery_rate",
        "single_mode_rate", "serial_mode_rate", "parallel_mode_rate"
    ]:
        ylabel += " (%)"
    plt.ylabel(ylabel)
    plt.title(f"HCRL convergence across learning rates: {metric}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = out_dir / f"hcrl_convergence_{metric}_wideaxis.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output")
    parser.add_argument("--out_dir", default="hcrl_png_convergence_figs_wideaxis")
    parser.add_argument("--tag_filter", default="")
    parser.add_argument("--smooth", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = discover_logs(args.input_dir, args.tag_filter)
    if not logs:
        raise FileNotFoundError(f"No txt logs found in {args.input_dir} with tag_filter={args.tag_filter!r}")

    frames = []
    for p in logs:
        df = parse_log(p)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No HCRL rows were parsed. Check your txt logs and run tags.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["lr", "seed", "episode"])
    df.to_csv(out_dir / "hcrl_convergence_all_episode_metrics.csv", index=False, encoding="utf-8-sig")

    metrics = [
        "reward",
        "success_rate",
        "success_time_rate",
        "malicious_rate",
        "trusted_rate",
        "cost_per_success",
        "backup_recovery_rate",
        "conditional_backup_recovery_rate",
        "single_mode_rate",
        "serial_mode_rate",
        "parallel_mode_rate",
    ]

    for metric in metrics:
        plot_metric(df, metric, out_dir, smooth=args.smooth)

    print("\n[OK] Wide-axis PNG convergence plots completed.")
    print(f"Parsed logs: {len(logs)}")
    print(f"Learning rates: {sorted(df['lr'].unique())}")
    print(f"Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
