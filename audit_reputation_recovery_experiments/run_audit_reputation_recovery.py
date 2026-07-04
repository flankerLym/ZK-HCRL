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

This patch extends the original behavior-level oracle attacks with six economic and
cross-chain adversarial scenarios:
    mev_based_manipulation, strategic_economic_collusion, cross_chain_latency_attack,
    real_world_oracle_cartel, bribery_staking_game, liquidation_front_running.
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
    "intermittent_evasion",
    "gradual_drift",
    "mev_based_manipulation",
    "strategic_economic_collusion",
    "cross_chain_latency_attack",
    "real_world_oracle_cartel",
    "bribery_staking_game",
    "liquidation_front_running",
]

LABELS = {
    "reputation_poisoning": "Reputation poisoning",
    "sleeper_attack": "Sleeper attack",
    "collusion_shift": "Collusion shift",
    "burst_attack": "Burst attack",
    "intermittent_evasion": "Intermittent evasion",
    "gradual_drift": "Gradual drift",
    "mev_based_manipulation": "MEV-based manipulation",
    "strategic_economic_collusion": "Strategic economic collusion",
    "cross_chain_latency_attack": "Cross-chain latency attack",
    "real_world_oracle_cartel": "Real-world oracle cartel",
    "bribery_staking_game": "Bribery / staking game",
    "liquidation_front_running": "Liquidation front-running",
}

SCENARIO_CATEGORY = {
    "reputation_poisoning": "behavioral",
    "sleeper_attack": "behavioral",
    "collusion_shift": "behavioral",
    "burst_attack": "behavioral",
    "intermittent_evasion": "behavioral",
    "gradual_drift": "behavioral",
    "mev_based_manipulation": "economic",
    "strategic_economic_collusion": "economic",
    "cross_chain_latency_attack": "cross_chain",
    "real_world_oracle_cartel": "economic_collusion",
    "bribery_staking_game": "economic",
    "liquidation_front_running": "mev_liquidation",
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
    max_audit_prob: float = 0.85


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
    p.add_argument("--requests", type=int, default=12000)
    p.add_argument("--oracles", type=int, default=120)
    p.add_argument("--malicious-ratio", type=float, default=0.30)
    p.add_argument("--trusted-ratio", type=float, default=0.35)
    p.add_argument("--interval", type=int, default=150)
    p.add_argument("--attack-onset-ratio", type=float, default=0.25)
    p.add_argument("--attack-end-ratio", type=float, default=0.65)
    p.add_argument("--scenarios", type=str, default=",".join(SCENARIOS),
                   help="Comma-separated scenario names. Use --list-scenarios to print all valid names.")
    p.add_argument("--list-scenarios", action="store_true", help="Print supported scenario names and exit.")
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


def high_value_wave(x: float, cycles: int = 8, power: int = 6) -> float:
    """Periodic high-value request window used by MEV/liquidation attacks."""
    x = float(np.clip(x, 0.0, 1.0))
    return float(max(0.0, math.sin(cycles * math.pi * x)) ** power)


def scenario_intensity(scenario: str, step: int, phase: PhaseConfig) -> float:
    ph = phase_for_step(step, phase)
    if ph == "benign":
        if scenario in {"reputation_poisoning", "bribery_staking_game"}:
            return 0.02
        if scenario == "sleeper_attack":
            return 0.00
        if scenario == "real_world_oracle_cartel":
            return 0.03
        return 0.04
    if ph == "recovery":
        return 0.015

    x = (step - phase.attack_onset) / max(phase.attack_end - phase.attack_onset, 1)

    # Original six behavior-level scenarios.
    if scenario == "reputation_poisoning":
        return 0.95
    if scenario == "sleeper_attack":
        return 1.00 if x > 0.05 else 0.75
    if scenario == "collusion_shift":
        return 0.88 + 0.08 * math.sin(10 * math.pi * x)
    if scenario == "burst_attack":
        return 0.40 if x < 0.25 else (1.00 if x < 0.75 else 0.55)
    if scenario == "intermittent_evasion":
        return 0.95 if int(x * 12) % 2 == 0 else 0.18
    if scenario == "gradual_drift":
        return 0.25 + 0.75 * x

    # New economic / MEV / cross-chain scenarios.
    if scenario == "mev_based_manipulation":
        # Attacks concentrate around high-value MEV opportunities.
        return 0.30 + 0.70 * high_value_wave(x, cycles=10, power=4)
    if scenario == "strategic_economic_collusion":
        # Several malicious operators coordinate with a slowly changing joint strategy.
        return 0.68 + 0.22 * (0.5 + 0.5 * math.sin(6 * math.pi * x))
    if scenario == "cross_chain_latency_attack":
        # Latency-based attacks are milder at first and worsen when bridge delay accumulates.
        return 0.35 + 0.60 * (x ** 0.7)
    if scenario == "real_world_oracle_cartel":
        # Cartel keeps attack persistent but not always maximal to avoid obvious detection.
        return 0.78 + 0.12 * math.sin(4 * math.pi * x)
    if scenario == "bribery_staking_game":
        # Stake/bribe pressure makes attack risk high after the first bidding window.
        return 0.55 + 0.35 * (1.0 if x > 0.18 else x / 0.18)
    if scenario == "liquidation_front_running":
        # Short liquidation-front-running bursts around several liquidation windows.
        burst = high_value_wave(x, cycles=12, power=8)
        return 0.22 + 0.78 * burst
    return 0.80


def scenario_selection_bonus(scenario: str, phase_name: str, is_malicious: np.ndarray, is_trusted: np.ndarray) -> np.ndarray:
    """Selection attractiveness bias before audit penalties become visible."""
    bonus = np.zeros_like(is_malicious, dtype=float)
    if phase_name == "benign":
        bonus[is_malicious] += 0.12
        bonus[is_trusted] += 0.08
    elif phase_name == "attack":
        bonus[is_malicious] += 0.05
        bonus[is_trusted] += 0.10
    else:
        bonus[is_trusted] += 0.10

    # Economic attacks can make adversarial nodes look more attractive through stake,
    # bribes or cartel-like routing before audit feedback suppresses them.
    if phase_name in {"benign", "attack"}:
        if scenario == "bribery_staking_game":
            bonus[is_malicious] += 0.14
        elif scenario == "strategic_economic_collusion":
            bonus[is_malicious] += 0.08
        elif scenario == "real_world_oracle_cartel":
            bonus[is_malicious] += 0.10
        elif scenario == "mev_based_manipulation":
            bonus[is_malicious] += 0.06
        elif scenario == "liquidation_front_running":
            bonus[is_malicious] += 0.07
    return bonus


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


def choose_oracle(eff: np.ndarray, malicious: np.ndarray, trusted: np.ndarray, phase_name: str,
                  scenario: str, rng: np.random.Generator) -> int:
    bonus = scenario_selection_bonus(scenario, phase_name, malicious, trusted)
    logits = 8.0 * (eff + bonus)
    logits -= np.max(logits)
    prob = np.exp(logits)
    prob = prob / np.sum(prob)
    return int(rng.choice(np.arange(len(eff)), p=prob))


def audit_risk(i: int, eff: np.ndarray, truth: np.ndarray, cooldown: np.ndarray, recent_fail: np.ndarray, params: AuditParams) -> float:
    cd = min(cooldown[i] / max(params.cooldown_steps, 1), 1.0)
    return float(np.clip(0.30 * (1 - eff[i]) + 0.30 * (1 - truth[i]) + 0.25 * recent_fail[i] + 0.15 * cd, 0.0, 1.0))


def scenario_failure_profile(scenario: str, intensity: float, ph: str, trisk: float) -> Tuple[float, float]:
    """Return additional failure probability and severity offset for a malicious node."""
    if ph != "attack":
        return 0.0, 0.0
    if scenario == "mev_based_manipulation":
        return 0.10 * intensity, 0.30 + 0.45 * intensity
    if scenario == "strategic_economic_collusion":
        return 0.08 + 0.08 * intensity, 0.35 + 0.35 * intensity
    if scenario == "cross_chain_latency_attack":
        return 0.18 + 0.18 * trisk, 0.55 + 0.55 * intensity
    if scenario == "real_world_oracle_cartel":
        return 0.13 + 0.10 * intensity, 0.55 + 0.50 * intensity
    if scenario == "bribery_staking_game":
        return 0.08 + 0.07 * intensity, 0.30 + 0.40 * intensity
    if scenario == "liquidation_front_running":
        return 0.12 * intensity, 0.65 + 0.75 * intensity
    return 0.0, 0.0


def simulate_attempt(scenario: str, is_mal: bool, is_tru: bool, intensity: float, ph: str,
                     trisk: float, rng: np.random.Generator) -> Tuple[bool, float, float]:
    """Return audit_pass, severity, validation_success_float."""
    if is_tru:
        fail_p = 0.020 + 0.030 * trisk
        severity_base = 0.25
    elif is_mal:
        if ph == "attack":
            extra_fail, extra_sev = scenario_failure_profile(scenario, intensity, ph, trisk)
            fail_p = 0.10 + 0.70 * intensity + 0.08 * trisk + extra_fail
            severity_base = 1.05 + 0.95 * intensity + 0.25 * trisk + extra_sev
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
        severity = float(np.clip(severity_base + rng.normal(0.0, 0.12), 0.5, 3.2))
        return False, severity, 0.0
    return True, 0.0, 1.0


def update_audit(i: int, passed: bool, severity: float, base: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                 clean: np.ndarray, cooldown: np.ndarray, params: AuditParams):
    if passed:
        alpha[i] += 1.0
        clean[i] += 1.0
        cooldown[i] = max(0.0, cooldown[i] - 1.0)
        if clean[i] >= params.min_clean_streak:
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
    if scenario in {"bribery_staking_game", "real_world_oracle_cartel"}:
        # Bribery/staking and cartel scenarios start with slightly better-looking attackers.
        base[malicious] += rng.normal(0.045, 0.010, malicious.sum())
    base = np.clip(base, 0.25, 0.82)

    alpha = np.full(args.oracles, params.alpha0, dtype=float)
    beta = np.full(args.oracles, params.beta0, dtype=float)
    clean = np.zeros(args.oracles, dtype=float)
    cooldown = np.zeros(args.oracles, dtype=float)
    recent_counts = np.zeros(args.oracles, dtype=float)
    recent_fail_counts = np.zeros(args.oracles, dtype=float)

    curve_rows = []
    event_rows = []

    for step in range(args.requests):
        ph = phase_for_step(step, phase)
        intensity = float(np.clip(scenario_intensity(scenario, step, phase), 0.0, 1.0))
        trisk = trace_risk(trace_df, step)
        cooldown = np.maximum(cooldown - 1.0, 0.0)
        eff, truth = effective_rep(base, alpha, beta, cooldown, params)
        recent_fail = recent_fail_counts / np.maximum(recent_counts, 1.0)
        selected = choose_oracle(eff, malicious, trusted, ph, scenario, rng)

        risk = audit_risk(selected, eff, truth, cooldown, recent_fail, params)
        p_audit = params.audit_base_rate + params.audit_risk_rate * risk
        if ph == "attack" and scenario in {"cross_chain_latency_attack", "liquidation_front_running", "real_world_oracle_cartel"}:
            p_audit += 0.05 + 0.05 * intensity
        if ph == "recovery" and malicious[selected]:
            p_audit += params.recovery_probe_rate
        p_audit = float(np.clip(p_audit, 0.0, params.max_audit_prob))
        audit_triggered = rng.random() < p_audit

        passed, sev, validation = simulate_attempt(scenario, bool(malicious[selected]), bool(trusted[selected]), intensity, ph, trisk, rng)
        if audit_triggered:
            update_audit(selected, passed, sev, base, alpha, beta, clean, cooldown, params)
            recent_counts[selected] += 1.0
            if not passed:
                recent_fail_counts[selected] += 1.0

        if step % args.interval == 0 or step == args.requests - 1:
            eff_now, truth_now = effective_rep(base, alpha, beta, cooldown, params)
            curve_rows.append({
                "scenario": scenario,
                "scenario_label": LABELS.get(scenario, scenario),
                "scenario_category": SCENARIO_CATEGORY.get(scenario, "unknown"),
                "seed": seed,
                "step": step,
                "phase": ph,
                "attack_intensity": intensity,
                "trace_risk": trisk,
                "malicious_rep_mean": float(np.mean(eff_now[malicious])),
                "trusted_rep_mean": float(np.mean(eff_now[trusted])),
                "normal_rep_mean": float(np.mean(eff_now[normal])),
                "malicious_truth_mean": float(np.mean(truth_now[malicious])),
                "trusted_truth_mean": float(np.mean(truth_now[trusted])),
                "reputation_gap": float(np.mean(eff_now[trusted]) - np.mean(eff_now[malicious])),
                "cooldown_malicious_mean": float(np.mean(cooldown[malicious])),
                "attack_onset_step": phase.attack_onset,
                "attack_end_step": phase.attack_end,
            })

        event_rows.append({
            "scenario": scenario,
            "scenario_label": LABELS.get(scenario, scenario),
            "scenario_category": SCENARIO_CATEGORY.get(scenario, "unknown"),
            "seed": seed,
            "step": step,
            "phase": ph,
            "selected_is_malicious": int(malicious[selected]),
            "selected_is_trusted": int(trusted[selected]),
            "audit_triggered": int(audit_triggered),
            "audit_pass": int(passed),
            "audit_failure": int(not passed),
            "severity": float(sev),
            "attack_intensity": float(intensity),
            "trace_risk": float(trisk),
            "selection_risk": float(risk),
            "validation_success": float(validation),
        })

    curve_df = pd.DataFrame(curve_rows)
    event_df = pd.DataFrame(event_rows)

    pre_start = max(0, phase.attack_onset - int(0.12 * args.requests))
    pre_end = phase.attack_onset
    attack_tail_start = max(phase.attack_onset, phase.attack_end - int(0.12 * args.requests))
    attack_tail_end = phase.attack_end
    rec_tail_start = max(phase.attack_end, args.requests - int(0.12 * args.requests))
    rec_tail_end = args.requests

    pre_rep = summarize_window(curve_df, pre_start, pre_end, "malicious_rep_mean")
    low_rep = summarize_window(curve_df, attack_tail_start, attack_tail_end, "malicious_rep_mean")
    rec_rep = summarize_window(curve_df, rec_tail_start, rec_tail_end, "malicious_rep_mean")
    trusted_end = summarize_window(curve_df, rec_tail_start, rec_tail_end, "trusted_rep_mean")
    gap_end = summarize_window(curve_df, rec_tail_start, rec_tail_end, "reputation_gap")
    drop_pct = 100.0 * (pre_rep - low_rep) / max(pre_rep, 1e-8)
    recovery_ratio = (rec_rep - low_rep) / max(pre_rep - low_rep, 1e-8)
    asymmetry = drop_pct / max(100.0 * max(rec_rep - low_rep, 0.0) / max(low_rep, 1e-8), 1e-8)

    attack_events = event_df[event_df["phase"] == "attack"]
    recovery_events = event_df[event_df["phase"] == "recovery"]
    summary = {
        "scenario": scenario,
        "scenario_label": LABELS.get(scenario, scenario),
        "scenario_category": SCENARIO_CATEGORY.get(scenario, "unknown"),
        "seed": seed,
        "requests": args.requests,
        "attack_onset_step": phase.attack_onset,
        "attack_end_step": phase.attack_end,
        "pre_attack_malicious_rep": pre_rep,
        "attack_end_malicious_rep": low_rep,
        "recovery_end_malicious_rep": rec_rep,
        "recovery_end_trusted_rep": trusted_end,
        "recovery_end_reputation_gap": gap_end,
        "malicious_rep_drop_pct": drop_pct,
        "malicious_rep_recovery_ratio": recovery_ratio,
        "asymmetry_score": asymmetry,
        "attack_malicious_selection_rate": float(attack_events["selected_is_malicious"].mean()) if len(attack_events) else float("nan"),
        "attack_audit_failure_rate": float(attack_events["audit_failure"].mean()) if len(attack_events) else float("nan"),
        "attack_mean_severity": float(attack_events.loc[attack_events["audit_failure"] == 1, "severity"].mean()) if len(attack_events) else float("nan"),
        "recovery_malicious_selection_rate": float(recovery_events["selected_is_malicious"].mean()) if len(recovery_events) else float("nan"),
        "recovery_audit_failure_rate": float(recovery_events["audit_failure"].mean()) if len(recovery_events) else float("nan"),
    }
    return curve_df, event_df, summary


def mean_std_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "pre_attack_malicious_rep",
        "attack_end_malicious_rep",
        "recovery_end_malicious_rep",
        "recovery_end_trusted_rep",
        "recovery_end_reputation_gap",
        "malicious_rep_drop_pct",
        "malicious_rep_recovery_ratio",
        "asymmetry_score",
        "attack_malicious_selection_rate",
        "attack_audit_failure_rate",
        "attack_mean_severity",
        "recovery_malicious_selection_rate",
        "recovery_audit_failure_rate",
    ]
    grouped = summary_df.groupby(["scenario", "scenario_label", "scenario_category"], as_index=False)
    rows = []
    for keys, df in grouped:
        scenario, scenario_label, scenario_category = keys
        row = {
            "scenario": scenario,
            "scenario_label": scenario_label,
            "scenario_category": scenario_category,
            "n_seeds": int(df["seed"].nunique()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(df[col].mean())
            row[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario_category", "scenario"]).reset_index(drop=True)


def write_paper_tables(out_dir: Path, stats: pd.DataFrame) -> None:
    paper_cols = [
        "scenario_label",
        "scenario_category",
        "pre_attack_malicious_rep_mean",
        "attack_end_malicious_rep_mean",
        "recovery_end_malicious_rep_mean",
        "malicious_rep_drop_pct_mean",
        "malicious_rep_recovery_ratio_mean",
        "asymmetry_score_mean",
        "attack_malicious_selection_rate_mean",
        "attack_audit_failure_rate_mean",
    ]
    paper = stats[paper_cols].copy()
    paper.to_csv(out_dir / "paper_table_reputation_recovery.csv", index=False)
    tex = paper.to_latex(index=False, float_format="%.3f")
    (out_dir / "paper_table_reputation_recovery.tex").write_text(tex, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.list_scenarios:
        print("Supported scenarios:")
        for s in SCENARIOS:
            print(f"  {s:<32} {LABELS[s]}")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_df = load_trace(args.trace)
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    requested_scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    unknown = [s for s in requested_scenarios if s not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Run with --list-scenarios for valid names.")

    all_curves: List[pd.DataFrame] = []
    all_events: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []

    print(f"[Info] running {len(requested_scenarios)} scenarios x {len(seeds)} seeds")
    for scenario in requested_scenarios:
        for seed in seeds:
            print(f"[Run] scenario={scenario}, seed={seed}")
            curve, events, summary = run_one(scenario, seed, args, trace_df)
            all_curves.append(curve)
            all_events.append(events)
            summaries.append(summary)

    curve_df = pd.concat(all_curves, ignore_index=True)
    event_df = pd.concat(all_events, ignore_index=True)
    summary_by_seed = pd.DataFrame(summaries)
    stats = mean_std_table(summary_by_seed)

    curve_df.to_csv(out_dir / "audit_reputation_recovery_curve.csv", index=False)
    event_df.to_csv(out_dir / "audit_reputation_recovery_event_timeline.csv", index=False)
    summary_by_seed.to_csv(out_dir / "audit_reputation_recovery_summary_by_seed.csv", index=False)
    stats.to_csv(out_dir / "audit_reputation_recovery_summary_mean_std.csv", index=False)
    write_paper_tables(out_dir, stats)

    print(f"[Done] wrote outputs to {out_dir}")
    print(f"[Done] scenarios: {', '.join(requested_scenarios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
