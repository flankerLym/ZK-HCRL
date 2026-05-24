from __future__ import annotations
import numpy as np
import pandas as pd

SCENARIOS = [
    "coordinated_shift",
    "reputation_poisoning_collusion",
    "latency_copattern",
    "failure_cooccurrence",
    "intermittent_evasion",
    "gradual_drift",
]

SCENARIO_LABELS = {
    "coordinated_shift": "Coordinated shift",
    "reputation_poisoning_collusion": "Reputation poisoning",
    "latency_copattern": "Latency co-pattern",
    "failure_cooccurrence": "Failure co-occurrence",
    "intermittent_evasion": "Intermittent evasion",
    "gradual_drift": "Gradual drift",
}


def assign_oracle_groups(n_oracles: int, malicious_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    ids = np.arange(n_oracles)
    n_mal = max(2, int(round(n_oracles * malicious_ratio)))
    malicious = rng.choice(ids, size=n_mal, replace=False)
    labels = np.zeros(n_oracles, dtype=int)
    labels[malicious] = 1
    service_type = ids % 3
    return labels, service_type


def phase_for_step(step: int, n_steps: int, onset: float, end: float) -> str:
    r = step / max(n_steps - 1, 1)
    if r < onset:
        return "benign"
    if r < end:
        return "attack"
    return "recovery"


def simulate_oracle_panel(
    trace: pd.DataFrame,
    scenario: str,
    seed: int,
    requests: int,
    n_oracles: int,
    malicious_ratio: float,
    attack_onset_ratio: float = 0.25,
    attack_end_ratio: float = 0.70,
) -> pd.DataFrame:
    """Simulate per-request oracle behavior using real trace as the market-risk driver."""
    rng = np.random.default_rng(seed)
    labels, service_type = assign_oracle_groups(n_oracles, malicious_ratio, seed)
    n_trace = len(trace)
    rep = np.full(n_oracles, 0.55, dtype=float)
    rows = []

    # Collusion group-specific latent signal creates correlated behavior.
    latent_common = rng.normal(0, 1, size=requests)

    for t in range(requests):
        tr = trace.iloc[t % n_trace]
        phase = phase_for_step(t, requests, attack_onset_ratio, attack_end_ratio)
        base_dev = float(tr.get("deviation", 0.001))
        base_stale = float(tr.get("staleness", 120.0))
        ref_success = float(tr.get("validation_success", 1.0))
        asset = str(tr.get("asset", "UNKNOWN"))
        common = latent_common[t]
        attack_progress = 0.0
        if phase == "attack":
            attack_progress = (t / requests - attack_onset_ratio) / max(attack_end_ratio - attack_onset_ratio, 1e-6)
            attack_progress = float(np.clip(attack_progress, 0, 1))

        for i in range(n_oracles):
            is_mal = labels[i] == 1
            noise = rng.normal(0, 0.0008)
            dev = base_dev * rng.lognormal(mean=0.0, sigma=0.20) + abs(noise)
            latency = base_stale * rng.lognormal(mean=0.0, sigma=0.25)
            fail_prob = np.clip(0.04 + 0.30 * base_dev + (1.0 - ref_success) * 0.12, 0.01, 0.35)
            response_bias = rng.normal(0, max(base_dev, 0.0005))
            audit_fail = 0

            if is_mal:
                if phase == "benign":
                    # Benign/camouflage behavior: looks slightly better than normal.
                    dev *= 0.75
                    latency *= 0.85
                    fail_prob *= 0.55
                    rep[i] += 0.003 * (1.0 - rep[i])
                elif phase == "attack":
                    if scenario == "coordinated_shift":
                        response_bias += 0.035 + 0.008 * common
                        dev += abs(0.018 + 0.006 * common)
                        fail_prob += 0.28
                    elif scenario == "reputation_poisoning_collusion":
                        response_bias += 0.040 + 0.006 * common
                        dev += abs(0.022 + 0.004 * common)
                        fail_prob += 0.33
                    elif scenario == "latency_copattern":
                        response_bias += 0.015 + 0.003 * common
                        dev += abs(0.010 + 0.002 * common)
                        latency += 550 + 90 * common
                        fail_prob += 0.22
                    elif scenario == "failure_cooccurrence":
                        burst = 1.0 if (t // 240) % 2 == 0 else 0.35
                        response_bias += burst * (0.025 + 0.004 * common)
                        dev += burst * abs(0.018 + 0.003 * common)
                        fail_prob += burst * 0.45
                    elif scenario == "intermittent_evasion":
                        active = 1.0 if (t // 360) % 3 != 1 else 0.0
                        response_bias += active * (0.032 + 0.006 * common)
                        dev += active * abs(0.018 + 0.004 * common)
                        fail_prob += active * 0.32
                    elif scenario == "gradual_drift":
                        drift = 0.006 + 0.040 * attack_progress
                        response_bias += drift + 0.003 * common
                        dev += abs(drift * 0.75 + 0.002 * common)
                        fail_prob += 0.14 + 0.18 * attack_progress
                    # Auditing and reputation update: fail quickly lowers reputation.
                    fail_prob = float(np.clip(fail_prob, 0.01, 0.92))
                    audit_fail = int(rng.random() < fail_prob)
                    if audit_fail:
                        rep[i] -= (0.020 + 0.035 * min(dev / 0.04, 1.0)) * rep[i]
                    else:
                        rep[i] += 0.002 * (1.0 - rep[i])
                else:  # recovery
                    dev *= 0.85
                    latency *= 0.95
                    fail_prob *= 0.55
                    audit_fail = int(rng.random() < fail_prob)
                    # Conservative recovery.
                    if audit_fail:
                        rep[i] -= 0.010 * rep[i]
                    else:
                        rep[i] += 0.004 * (1.0 - rep[i])
            else:
                if phase == "attack":
                    # Benign nodes may be affected by market risk but do not share collusive signal.
                    dev *= rng.lognormal(0, 0.08)
                    latency *= rng.lognormal(0, 0.08)
                audit_fail = int(rng.random() < fail_prob)
                if audit_fail:
                    rep[i] -= 0.004 * rep[i]
                else:
                    rep[i] += 0.0025 * (1.0 - rep[i])

            rep[i] = float(np.clip(rep[i], 0.02, 0.99))
            response = 1.0 + response_bias + rng.normal(0, max(dev * 0.25, 0.0005))
            rows.append({
                "step": t,
                "oracle_id": i,
                "scenario": scenario,
                "phase": phase,
                "asset": asset,
                "service_type": int(service_type[i]),
                "is_colluder": int(is_mal),
                "deviation": float(max(dev, 0.0)),
                "latency": float(max(latency, 0.0)),
                "validation_fail": int(audit_fail),
                "response_norm": float(response),
                "reputation": float(rep[i]),
            })

    return pd.DataFrame(rows)
