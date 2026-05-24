"""Audit reputation degradation-and-recovery experiment.

This script is self-contained and designed to be placed under the TCO-DRL root as:
    audit_reputation_recovery_experiments/run_audit_reputation_recovery.py

It isolates the audit-aware reputation dynamics from HCRL and simulates long request
sequences with three phases:
    benign/camouflage -> attack-active -> recovery/benign

The update rule follows the HCRL audit design:
    - audit failure: beta += severity, clean_streak = 0, base reputation decreases fast;
    - audit pass: alpha += 1, clean_streak += 1, cooldown decays;
    - after clean_streak threshold: base reputation recovers slowly by pass_recovery*(1-base);
    - effective reputation = (1-w)*base + w*truth_score - cooldown_penalty.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCENARIOS = [
    "reputation_poisoning",
    "sleeper_attack",
    "collusion_shift",
    "burst_attack",
    "gradual_drift",
    "intermittent_evasion",
]

LABELS = {
    "reputation_poisoning": "Reputation poisoning",
    "sleeper_attack": "Sleeper attack",
    "collusion_shift": "Collusion shift",
    "burst_attack": "Burst attack",
    "gradual_drift": "Gradual drift",
    "intermittent_evasion": "Intermittent evasion",
}


@dataclass
class AuditParams:
    alpha0: float = 2.0
    beta0: float = 2.0
    audit_weight: float = 0.30
    cooldown_penalty: float = 0.12
    cooldown_steps: int = 450
    min_clean_streak: int = 3
    pass_recovery: float = 0.018  # intentionally slow recovery
    fail_penalty: float = 0.080
    audit_base_rate: float = 0.050
    audit_risk_rate: float = 0.340
    recovery_probe_rate: float = 0.180
    max_audit_prob: float = 0.80


@dataclass
class PhaseConfig:
    attack_onset: int
    attack_end: int
    requests: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run audit reputation degradation-and-recovery experiments.")
    p.add_argument("--trace", type=str, default="experiments_real_trace/data/real_oracle_trace.csv",
                   help="Optional real trace CSV. Used only as background anomaly/risk variability if present.")
    p.add_argument("--out", type=str, default="audit_reputation_recovery_experiments/output")
    p.add_argument("--seeds", type=str, default="3,4,5,6,7")
    p.add_argument("--requests", type=int, default=9000)
    p.add_argument("--oracles", type=int, default=120)
    p.add_argument("--malicious-ratio", type=float, default=0.30)
    p.add_argument("--trusted-ratio", type=float, default=0.35)
    p.add_argument("--interval", type=int, default=150)
    p.add_argument("--attack-onset-ratio", type=float, default=0.33)
    p.add_argument("--attack-end-ratio", type=float, default=0.56)
    p.add_argument("--scenarios", type=str, default=",".join(SCENARIOS),
                   help="Comma-separated scenario names.")
    p.add_argument("--pass-recovery", type=float, default=AuditParams.pass_recovery,
                   help="Slow reputation recovery coefficient after clean streak.")
    p.add_argument("--fail-penalty", type=float, default=AuditParams.fail_penalty,
                   help="Fast severity-weighted reputation penalty on audit failure.")
    p.add_argument("--recovery-probe-rate", type=float, default=AuditParams.recovery_probe_rate,
                   help="Extra audit/probe probability for previously malicious nodes during recovery.")
    return p.parse_args()


def load_trace(trace_path: str) -> pd.DataFrame | None:
    path = Path(trace_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    for col in ["deviation", "staleness", "validation_success"]:
        if col not in df.columns:
            df[col] = 0.0
    df["deviation"] = pd.to_numeric(df["deviation"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["staleness"] = pd.to_numeric(df["staleness"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["validation_success"] = pd.to_numeric(df["validation_success"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    if "anomaly_label" not in df.columns:
        df["anomaly_label"] = "normal"
    df["anomaly_label"] = df["anomaly_label"].astype(str).str.lower()
    return df.reset_index(drop=True)


def trace_risk(trace_df: pd.DataFrame | None, step: int) -> float:
    if trace_df is None or len(trace_df) == 0:
        return 0.0
    row = trace_df.iloc[step % len(trace_df)]
    label = str(row.get("anomaly_label", "normal")).lower()
    label_risk = 0.0
    if "anomal" in label:
        label_risk = 0.7
    elif "susp" in label:
        label_risk = 0.35
    deviation = min(float(row.get("deviation", 0.0)) / 0.02, 1.0)
    stale = min(float(row.get("staleness", 0.0)) / 7200.0, 1.0)
    val_fail = 1.0 - float(row.get("validation_success", 1.0))
    return float(np.clip(0.45 * label_risk + 0.25 * deviation + 0.20 * stale + 0.10 * val_fail, 0.0, 1.0))


def phase_for_step(step: int, phase: PhaseConfig) -> str:
    if step < phase.attack_onset:
        return "benign"
    if step < phase.attack_end:
        return "attack"
    return "recovery"


def scenario_intensity(scenario: str, step: int, phase: PhaseConfig) -> float:
    ph = phase_for_step(step, phase)
    if ph == "benign":
        # Camouflage stage: malicious nodes mostly behave well.
        if scenario == "reputation_poisoning":
            return 0.02
        if scenario == "sleeper_attack":
            return 0.00
        return 0.04
    if ph == "recovery":
        # Recovery stage: malicious nodes stop attacking and keep benign behavior.
        return 0.015

    # Attack-active stage.
    x = (step - phase.attack_onset) / max(phase.attack_end - phase.attack_onset, 1)
    if scenario == "reputation_poisoning":
        return 0.95
    if scenario == "sleeper_attack":
        return 1.00 if x > 0.05 else 0.75
    if scenario == "collusion_shift":
        return 0.88 + 0.08 * math.sin(10 * math.pi * x)
    if scenario == "burst_attack":
        return 0.40 if x < 0.25 else (1.00 if x < 0.75 else 0.55)
    if scenario == "gradual_drift":
        return 0.25 + 0.75 * x
    if scenario == "intermittent_evasion":
        return 0.95 if int(x * 12) % 2 == 0 else 0.18
    return 0.80


def init_oracles(n_oracles: int, malicious_ratio: float, trusted_ratio: float, rng: np.random.Generator):
    n_mal = max(1, int(round(n_oracles * malicious_ratio)))
    n_trusted = max(1, int(round(n_oracles * trusted_ratio)))
    n_trusted = min(n_trusted, n_oracles - n_mal)
    idx = np.arange(n_oracles)
    rng.shuffle(idx)
    malicious = np.zeros(n_oracles, dtype=bool)
    trusted = np.zeros(n_oracles, dtype=bool)
    malicious[idx[:n_mal]] = True
    trusted[idx[n_mal:n_mal + n_trusted]] = True
    normal = ~(malicious | trusted)
    return malicious, trusted, normal


def effective_rep(base: np.ndarray, alpha: np.ndarray, beta: np.ndarray, cooldown: np.ndarray, params: AuditParams) -> Tuple[np.ndarray, np.ndarray]:
    truth = alpha / np.maximum(alpha + beta, 1e-8)
    cd_frac = np.clip(cooldown / max(params.cooldown_steps, 1), 0.0, 1.0)
    rep = np.clip((1.0 - params.audit_weight) * base + params.audit_weight * truth - params.cooldown_penalty * cd_frac, 0.0, 1.0)
    return rep, truth


def choose_oracle(eff: np.ndarray, malicious: np.ndarray, trusted: np.ndarray, phase_name: str, rng: np.random.Generator) -> int:
    # Attractiveness term lets malicious nodes be selected in camouflage/early attack,
    # while lower reputation naturally suppresses them after being penalized.
    bonus = np.zeros_like(eff)
    if phase_name == "benign":
        bonus[malicious] += 0.12
        bonus[trusted] += 0.08
    elif phase_name == "attack":
        bonus[malicious] += 0.05
        bonus[trusted] += 0.10
    else:
        bonus[trusted] += 0.10
        bonus[malicious] += 0.00
    logits = 8.0 * (eff + bonus)
    logits -= np.max(logits)
    prob = np.exp(logits)
    prob = prob / np.sum(prob)
    return int(rng.choice(np.arange(len(eff)), p=prob))


def audit_risk(i: int, eff: np.ndarray, truth: np.ndarray, cooldown: np.ndarray, recent_fail: np.ndarray, params: AuditParams) -> float:
    cd = min(cooldown[i] / max(params.cooldown_steps, 1), 1.0)
    return float(np.clip(0.30 * (1 - eff[i]) + 0.30 * (1 - truth[i]) + 0.25 * recent_fail[i] + 0.15 * cd, 0.0, 1.0))


def simulate_attempt(is_mal: bool, is_tru: bool, intensity: float, ph: str, trisk: float, rng: np.random.Generator) -> Tuple[bool, float, float]:
    """Return audit_pass, severity, validation_success_float."""
    if is_tru:
        fail_p = 0.020 + 0.030 * trisk
        severity_base = 0.25
    elif is_mal:
        if ph == "attack":
            fail_p = 0.10 + 0.82 * intensity + 0.08 * trisk
            severity_base = 1.20 + 1.10 * intensity + 0.25 * trisk
        elif ph == "recovery":
            fail_p = 0.025 + 0.035 * trisk
            severity_base = 0.35
        else:
            fail_p = 0.025 + 0.025 * trisk
            severity_base = 0.30
    else:
        fail_p = 0.055 + 0.080 * trisk
        severity_base = 0.45
    fail_p = float(np.clip(fail_p, 0.0, 0.98))
    fail = rng.random() < fail_p
    if fail:
        severity = float(np.clip(severity_base + rng.normal(0.0, 0.12), 0.5, 2.5))
        return False, severity, 0.0
    return True, 0.0, 1.0


def update_audit(i: int, passed: bool, severity: float, base: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                 clean: np.ndarray, cooldown: np.ndarray, params: AuditParams):
    if passed:
        alpha[i] += 1.0
        clean[i] += 1.0
        cooldown[i] = max(0.0, cooldown[i] - 1.0)
        if clean[i] >= params.min_clean_streak:
            # Conservative recovery: closer to 1, but very slowly.
            base[i] += params.pass_recovery * (1.0 - base[i])
    else:
        sev = max(float(severity), 0.5)
        beta[i] += sev
        clean[i] = 0.0
        base[i] -= params.fail_penalty * sev
        if sev >= 1.5:
            cooldown[i] = float(params.cooldown_steps)
    base[i] = float(np.clip(base[i], 0.0, 1.0))


def summarize_window(df: pd.DataFrame, start: int, end: int, col: str) -> float:
    s = df[(df["step"] >= start) & (df["step"] < end)][col]
    return float(s.mean()) if len(s) else float("nan")


def run_one(scenario: str, seed: int, args: argparse.Namespace, trace_df: pd.DataFrame | None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    params = AuditParams(pass_recovery=args.pass_recovery,
                         fail_penalty=args.fail_penalty,
                         recovery_probe_rate=args.recovery_probe_rate)
    phase = PhaseConfig(
        attack_onset=int(round(args.requests * args.attack_onset_ratio)),
        attack_end=int(round(args.requests * args.attack_end_ratio)),
        requests=args.requests,
    )
    malicious, trusted, normal = init_oracles(args.oracles, args.malicious_ratio, args.trusted_ratio, rng)

    base = np.zeros(args.oracles, dtype=float)
    base[trusted] = rng.normal(0.66, 0.025, trusted.sum())
    base[normal] = rng.normal(0.52, 0.035, normal.sum())
    base[malicious] = rng.normal(0.56, 0.030, malicious.sum())
    base = np.clip(base, 0.25, 0.80)
    alpha = np.full(args.oracles, params.alpha0, dtype=float)
    beta = np.full(args.oracles, params.beta0, dtype=float)
    clean = np.zeros(args.oracles, dtype=float)
    cooldown = np.zeros(args.oracles, dtype=float)
    recent_counts = np.zeros(args.oracles, dtype=float)
    recent_fail_counts = np.zeros(args.oracles, dtype=float)

    curve_rows = []
    event_rows = []
    acc = []

    for step in range(args.requests):
        ph = phase_for_step(step, phase)
        intensity = float(np.clip(scenario_intensity(scenario, step, phase), 0.0, 1.0))
        trisk = trace_risk(trace_df, step)
        cooldown = np.maximum(cooldown - 1.0, 0.0)
        eff, truth = effective_rep(base, alpha, beta, cooldown, params)
        recent_fail = recent_fail_counts / np.maximum(recent_counts, 1.0)
        selected = choose_oracle(eff, malicious, trusted, ph, rng)

        risk = audit_risk(selected, eff, truth, cooldown, recent_fail, params)
        p_audit = params.audit_base_rate + params.audit_risk_rate * risk
        if ph == "recovery" and malicious[selected]:
            p_audit += params.recovery_probe_rate
        p_audit = float(np.clip(p_audit, 0.0, params.max_audit_prob))
        audit_triggered = rng.random() < p_audit

        passed, sev, validation = simulate_attempt(bool(malicious[selected]), bool(trusted[selected]), intensity, ph, trisk, rng)
        if audit_triggered:
            update_audit(selected, passed, sev, base, alpha, beta, clean, cooldown, params)
            recent_counts[selected] += 1.0
            recent_fail_counts[selected] += 0.0 if passed else 1.0

        # Recovery probes: even if malicious nodes are no longer selected, sampled checks allow slow restoration
        # under sustained good behavior. This is analogous to periodic audit/re-evaluation.
        if ph == "recovery" and rng.random() < params.recovery_probe_rate:
            candidates = np.where(malicious)[0]
            if candidates.size > 0:
                probe = int(rng.choice(candidates))
                p2, s2, _ = simulate_attempt(True, False, 0.015, ph, trisk, rng)
                update_audit(probe, p2, s2, base, alpha, beta, clean, cooldown, params)
                recent_counts[probe] += 1.0
                recent_fail_counts[probe] += 0.0 if p2 else 1.0

        eff_after, truth_after = effective_rep(base, alpha, beta, cooldown, params)
        acc.append({
            "selected_malicious": float(malicious[selected]),
            "selected_trusted": float(trusted[selected]),
            "audit_triggered": float(audit_triggered),
            "audit_failed": float(audit_triggered and not passed),
            "success": float(validation),
            "attack_intensity": intensity,
        })

        event_rows.append({
            "scenario": scenario,
            "seed": seed,
            "step": step,
            "phase": ph,
            "oracle_id": selected,
            "selected_malicious": int(malicious[selected]),
            "selected_trusted": int(trusted[selected]),
            "audit_triggered": int(audit_triggered),
            "audit_failed": int(audit_triggered and not passed),
            "audit_probability": p_audit,
            "audit_risk": risk,
            "validation_success": validation,
            "attack_intensity": intensity,
            "severity": sev,
            "attack_onset_step": phase.attack_onset,
            "attack_end_step": phase.attack_end,
        })

        if step % args.interval == 0 or step == args.requests - 1:
            window = acc[-args.interval:] if acc else []
            def wm(k):
                return float(np.mean([r[k] for r in window])) if window else 0.0
            curve_rows.append({
                "scenario": scenario,
                "seed": seed,
                "step": step,
                "phase": ph,
                "malicious_rep_mean": float(np.mean(eff_after[malicious])),
                "trusted_rep_mean": float(np.mean(eff_after[trusted])),
                "normal_rep_mean": float(np.mean(eff_after[normal])) if normal.any() else np.nan,
                "malicious_base_rep_mean": float(np.mean(base[malicious])),
                "trusted_base_rep_mean": float(np.mean(base[trusted])),
                "audit_truth_malicious_mean": float(np.mean(truth_after[malicious])),
                "audit_truth_trusted_mean": float(np.mean(truth_after[trusted])),
                "reputation_gap": float(np.mean(eff_after[trusted]) - np.mean(eff_after[malicious])),
                "audit_rate": wm("audit_triggered"),
                "audit_fail_rate": wm("audit_failed"),
                "selected_malicious_rate": wm("selected_malicious"),
                "selected_trusted_rate": wm("selected_trusted"),
                "success_rate": wm("success"),
                "attack_intensity_mean": wm("attack_intensity"),
                "attack_onset_step": phase.attack_onset,
                "attack_end_step": phase.attack_end,
            })

    curve = pd.DataFrame(curve_rows)
    events = pd.DataFrame(event_rows)

    # windows around key points
    w = max(args.interval * 2, 1)
    pre_rep = summarize_window(curve, max(0, phase.attack_onset - 3 * w), phase.attack_onset, "malicious_rep_mean")
    attack_end_rep = summarize_window(curve, max(phase.attack_onset, phase.attack_end - 2 * w), phase.attack_end + 1, "malicious_rep_mean")
    recovery_end_rep = summarize_window(curve, max(phase.attack_end, args.requests - 3 * w), args.requests + 1, "malicious_rep_mean")
    pre_truth = summarize_window(curve, max(0, phase.attack_onset - 3 * w), phase.attack_onset, "audit_truth_malicious_mean")
    attack_end_truth = summarize_window(curve, max(phase.attack_onset, phase.attack_end - 2 * w), phase.attack_end + 1, "audit_truth_malicious_mean")
    recovery_end_truth = summarize_window(curve, max(phase.attack_end, args.requests - 3 * w), args.requests + 1, "audit_truth_malicious_mean")
    pre_gap = summarize_window(curve, max(0, phase.attack_onset - 3 * w), phase.attack_onset, "reputation_gap")
    attack_gap = summarize_window(curve, max(phase.attack_onset, phase.attack_end - 2 * w), phase.attack_end + 1, "reputation_gap")
    recovery_gap = summarize_window(curve, max(phase.attack_end, args.requests - 3 * w), args.requests + 1, "reputation_gap")

    drop_abs = max(pre_rep - attack_end_rep, 0.0)
    recovery_abs = max(recovery_end_rep - attack_end_rep, 0.0)
    drop_pct = 100.0 * drop_abs / max(pre_rep, 1e-8)
    recovery_ratio = recovery_abs / max(drop_abs, 1e-8)

    attack_curve = curve[(curve["step"] >= phase.attack_onset) & (curve["step"] <= phase.attack_end)]
    recovery_curve = curve[curve["step"] >= phase.attack_end]
    drop_threshold = pre_rep - 0.50 * drop_abs
    recover_threshold = attack_end_rep + 0.25 * drop_abs
    drop_lag = np.nan
    if len(attack_curve):
        idx = attack_curve[attack_curve["malicious_rep_mean"] <= drop_threshold]
        if len(idx):
            drop_lag = float((idx.iloc[0]["step"] - phase.attack_onset) / max(args.interval, 1))
    recovery_lag = np.nan
    if len(recovery_curve):
        idx = recovery_curve[recovery_curve["malicious_rep_mean"] >= recover_threshold]
        if len(idx):
            recovery_lag = float((idx.iloc[0]["step"] - phase.attack_end) / max(args.interval, 1))

    attack_dur = max(phase.attack_end - phase.attack_onset, 1)
    recovery_dur = max(args.requests - phase.attack_end, 1)
    drop_slope = drop_abs / attack_dur
    recovery_slope = recovery_abs / recovery_dur
    asymmetry = drop_slope / max(recovery_slope, 1e-10)

    summary = {
        "scenario": scenario,
        "seed": seed,
        "requests": args.requests,
        "interval": args.interval,
        "attack_onset_step": phase.attack_onset,
        "attack_end_step": phase.attack_end,
        "pre_attack_malicious_rep": pre_rep,
        "attack_end_malicious_rep": attack_end_rep,
        "recovery_end_malicious_rep": recovery_end_rep,
        "malicious_rep_drop_abs": drop_abs,
        "malicious_rep_drop_pct": drop_pct,
        "malicious_rep_recovery_abs": recovery_abs,
        "malicious_rep_recovery_ratio": recovery_ratio,
        "pre_attack_malicious_truth": pre_truth,
        "attack_end_malicious_truth": attack_end_truth,
        "recovery_end_malicious_truth": recovery_end_truth,
        "truth_drop_abs": max(pre_truth - attack_end_truth, 0.0),
        "truth_recovery_abs": max(recovery_end_truth - attack_end_truth, 0.0),
        "pre_attack_reputation_gap": pre_gap,
        "attack_end_reputation_gap": attack_gap,
        "recovery_end_reputation_gap": recovery_gap,
        "gap_increase_after_attack": attack_gap - pre_gap,
        "gap_after_recovery": recovery_gap,
        "drop_lag_intervals": drop_lag,
        "recovery_lag_intervals": recovery_lag,
        "drop_slope_per_request": drop_slope,
        "recovery_slope_per_request": recovery_slope,
        "asymmetry_score": asymmetry,
        "attack_audit_rate": summarize_window(curve, phase.attack_onset, phase.attack_end, "audit_rate"),
        "attack_audit_fail_rate": summarize_window(curve, phase.attack_onset, phase.attack_end, "audit_fail_rate"),
        "recovery_audit_rate": summarize_window(curve, phase.attack_end, args.requests, "audit_rate"),
        "recovery_audit_fail_rate": summarize_window(curve, phase.attack_end, args.requests, "audit_fail_rate"),
        "attack_selected_malicious_rate": summarize_window(curve, phase.attack_onset, phase.attack_end, "selected_malicious_rate"),
        "recovery_selected_malicious_rate": summarize_window(curve, phase.attack_end, args.requests, "selected_malicious_rate"),
        "degradation_success": float(drop_pct > 50.0),
        "recovery_success": float(recovery_abs > 0.02 and recovery_ratio > 0.05),
    }
    return curve, events, summary


def mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in ["scenario"] and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for scenario, g in df.groupby("scenario", sort=True):
        row = {"scenario": scenario}
        for c in numeric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def make_latex_table(df: pd.DataFrame, out_path: Path):
    cols = [
        "scenario",
        "pre_attack_malicious_rep_mean",
        "attack_end_malicious_rep_mean",
        "recovery_end_malicious_rep_mean",
        "malicious_rep_drop_pct_mean",
        "malicious_rep_recovery_ratio_mean",
        "asymmetry_score_mean",
        "attack_audit_rate_mean",
        "recovery_audit_rate_mean",
    ]
    available = [c for c in cols if c in df.columns]
    tex_df = df[available].copy()
    tex_df["scenario"] = tex_df["scenario"].map(lambda s: LABELS.get(s, str(s).replace("_", " ").title()))
    for c in tex_df.columns:
        if c != "scenario":
            tex_df[c] = tex_df[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_df.to_latex(index=False, escape=False))


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    trace_df = load_trace(args.trace)
    if trace_df is None:
        print(f"[Info] real trace not found or unreadable: {args.trace}. Running with synthetic background variability.")
    else:
        print(f"[Info] loaded real trace: {args.trace}, rows={len(trace_df)}")

    curves = []
    events = []
    summaries = []
    for scenario in scenarios:
        for seed in seeds:
            print(f"[Run] scenario={scenario}, seed={seed}")
            c, e, s = run_one(scenario, seed, args, trace_df)
            curves.append(c)
            events.append(e)
            summaries.append(s)

    curve_df = pd.concat(curves, ignore_index=True)
    event_df = pd.concat(events, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    meanstd_df = mean_std_table(summary_df)

    curve_df.to_csv(out / "audit_reputation_recovery_curve.csv", index=False)
    event_df.to_csv(out / "audit_reputation_recovery_event_timeline.csv", index=False)
    summary_df.to_csv(out / "audit_reputation_recovery_summary_by_seed.csv", index=False)
    meanstd_df.to_csv(out / "audit_reputation_recovery_summary_mean_std.csv", index=False)

    paper_cols = [
        "scenario",
        "pre_attack_malicious_rep_mean",
        "attack_end_malicious_rep_mean",
        "recovery_end_malicious_rep_mean",
        "malicious_rep_drop_abs_mean",
        "malicious_rep_drop_pct_mean",
        "malicious_rep_recovery_abs_mean",
        "malicious_rep_recovery_ratio_mean",
        "truth_drop_abs_mean",
        "truth_recovery_abs_mean",
        "gap_increase_after_attack_mean",
        "gap_after_recovery_mean",
        "drop_lag_intervals_mean",
        "recovery_lag_intervals_mean",
        "asymmetry_score_mean",
        "attack_audit_rate_mean",
        "attack_audit_fail_rate_mean",
        "recovery_audit_rate_mean",
        "recovery_audit_fail_rate_mean",
        "degradation_success_mean",
        "recovery_success_mean",
    ]
    paper_cols = [c for c in paper_cols if c in meanstd_df.columns]
    paper = meanstd_df[paper_cols].copy()
    paper.to_csv(out / "paper_table_reputation_recovery.csv", index=False)
    make_latex_table(meanstd_df, out / "paper_table_reputation_recovery.tex")

    print(f"[Done] outputs written to {out}")
    with pd.option_context("display.max_columns", 200, "display.width", 220):
        print(paper)


if __name__ == "__main__":
    main()
