"""Dynamic attack scenarios for reputation-drop validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

import numpy as np


@dataclass(frozen=True)
class AttackScenario:
    name: str
    description: str
    onset: float = 0.30
    end: float = 1.00
    strength: float = 1.0
    collusion_fraction: float = 0.60


def default_scenarios() -> List[AttackScenario]:
    return [
        AttackScenario("static_malicious", "Malicious oracles attack throughout the horizon.", onset=0.00, end=1.00, strength=0.85),
        AttackScenario("reputation_poisoning", "Malicious oracles behave cleanly early and attack after reputation is accumulated.", onset=0.30, end=1.00, strength=1.00),
        AttackScenario("burst_attack", "Attack intensity sharply increases in a finite middle window.", onset=0.40, end=0.65, strength=1.00),
        AttackScenario("sleeper_attack", "Malicious oracles sleep for half of the horizon and then attack aggressively.", onset=0.50, end=1.00, strength=1.00),
        AttackScenario("collusion_shift", "A colluding subset of malicious oracles jointly shifts behavior after warm-up.", onset=0.25, end=1.00, strength=1.00, collusion_fraction=0.65),
        AttackScenario("gradual_drift", "Malicious behavior gradually increases after warm-up, simulating slow drift.", onset=0.25, end=1.00, strength=1.00),
        AttackScenario("intermittent_evasion", "Malicious oracles alternate short attack and clean intervals to evade auditing.", onset=0.25, end=1.00, strength=1.00),
    ]


def scenario_by_names(names: Iterable[str] | None) -> List[AttackScenario]:
    scenarios = default_scenarios()
    if not names:
        return scenarios
    lookup = {s.name: s for s in scenarios}
    out = []
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        if name not in lookup:
            raise ValueError("Unknown scenario '%s'. Available: %s" % (name, sorted(lookup)))
        out.append(lookup[name])
    return out


def attack_intensity(scenario: AttackScenario, step: int, total_steps: int, is_malicious: bool, oracle_id: int, colluder_ids: Set[int], trace_risk: float) -> float:
    if not is_malicious:
        return 0.0
    phase = float(step) / max(float(total_steps), 1.0)
    tr = float(np.clip(trace_risk, 0.0, 1.0))

    if scenario.name == "static_malicious":
        return float(np.clip(0.55 + 0.35 * tr, 0.0, scenario.strength))
    if scenario.name == "reputation_poisoning":
        if phase < scenario.onset:
            return float(np.clip(0.02 + 0.03 * tr, 0.0, 0.08))
        return float(np.clip(0.70 + 0.30 * tr, 0.0, scenario.strength))
    if scenario.name == "burst_attack":
        if scenario.onset <= phase <= scenario.end:
            return float(np.clip(0.78 + 0.22 * tr, 0.0, scenario.strength))
        return float(np.clip(0.06 + 0.06 * tr, 0.0, 0.16))
    if scenario.name == "sleeper_attack":
        if phase < scenario.onset:
            return float(np.clip(0.01 + 0.03 * tr, 0.0, 0.06))
        return float(np.clip(0.76 + 0.24 * tr, 0.0, scenario.strength))
    if scenario.name == "collusion_shift":
        if phase < scenario.onset:
            return float(np.clip(0.03 + 0.04 * tr, 0.0, 0.10))
        if int(oracle_id) in colluder_ids:
            return float(np.clip(0.82 + 0.18 * tr, 0.0, scenario.strength))
        return float(np.clip(0.35 + 0.25 * tr, 0.0, 0.65))
    if scenario.name == "gradual_drift":
        if phase < scenario.onset:
            return float(np.clip(0.03 + 0.04 * tr, 0.0, 0.10))
        progress = (phase - scenario.onset) / max(1.0 - scenario.onset, 1e-8)
        return float(np.clip(0.10 + 0.80 * progress + 0.10 * tr, 0.0, scenario.strength))
    if scenario.name == "intermittent_evasion":
        if phase < scenario.onset:
            return float(np.clip(0.02 + 0.03 * tr, 0.0, 0.08))
        local = int((phase - scenario.onset) * 100)
        attacking = (local % 12) < 7
        return float(np.clip((0.78 if attacking else 0.06) + 0.18 * tr, 0.0, scenario.strength if attacking else 0.18))
    return float(np.clip(0.50 + 0.30 * tr, 0.0, scenario.strength))


def phase_label(scenario: AttackScenario, step: int, total_steps: int) -> str:
    phase = float(step) / max(float(total_steps), 1.0)
    if scenario.name == "static_malicious":
        return "attack"
    if phase < scenario.onset:
        return "pre_attack"
    if scenario.onset <= phase <= scenario.end:
        return "attack"
    return "post_attack"


def attack_markers(scenario: AttackScenario, total_steps: int) -> Dict[str, int]:
    return {
        "attack_onset_step": int(round(scenario.onset * total_steps)),
        "attack_end_step": int(round(scenario.end * total_steps)),
    }
