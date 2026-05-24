"""Standalone HCRL-style audit reputation module.

This module extracts only the audit-aware reputation idea: Beta posterior truth
score, risk-triggered audit probability, asymmetric penalty/recovery, and
cooldown. It does not depend on RL training or the original TCO-DRL loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class AuditConfig:
    alpha0: float = 2.0
    beta0: float = 2.0
    base_rate: float = 0.08
    risk_rate: float = 0.45
    max_rate: float = 0.80
    weight_in_reputation: float = 0.35
    cooldown_steps: int = 300
    cooldown_penalty: float = 0.12
    fail_penalty: float = 0.18
    pass_recovery: float = 0.025
    min_clean_streak: int = 3
    base_success_recovery: float = 0.012
    base_fail_penalty: float = 0.070


class AuditReputation:
    def __init__(self, n_oracles: int, cfg: Optional[AuditConfig] = None):
        self.n_oracles = int(n_oracles)
        self.cfg = cfg or AuditConfig()
        self.alpha = np.full(self.n_oracles, self.cfg.alpha0, dtype=float)
        self.beta = np.full(self.n_oracles, self.cfg.beta0, dtype=float)
        self.clean_streak = np.zeros(self.n_oracles, dtype=float)
        self.cooldown = np.zeros(self.n_oracles, dtype=float)
        self.last_step = np.full(self.n_oracles, -1.0, dtype=float)
        self.pass_count = np.zeros(self.n_oracles, dtype=float)
        self.fail_count = np.zeros(self.n_oracles, dtype=float)
        self.trigger_count = np.zeros(self.n_oracles, dtype=float)

    def tick(self) -> None:
        self.cooldown = np.maximum(self.cooldown - 1.0, 0.0)

    def truth_score(self) -> np.ndarray:
        return self.alpha / np.maximum(self.alpha + self.beta, 1e-8)

    def cooldown_fraction(self) -> np.ndarray:
        return np.clip(self.cooldown / max(float(self.cfg.cooldown_steps), 1.0), 0.0, 1.0)

    def effective_reputation(self, base_rep: np.ndarray) -> np.ndarray:
        base = np.clip(np.asarray(base_rep, dtype=float), 0.0, 1.0)
        truth = self.truth_score()
        penalty = self.cfg.cooldown_penalty * self.cooldown_fraction()
        return np.clip((1.0 - self.cfg.weight_in_reputation) * base + self.cfg.weight_in_reputation * truth - penalty, 0.0, 1.0)

    def audit_risk(self, oracle_id: int, base_rep: np.ndarray, recent_fail_rate: np.ndarray, global_step: int) -> float:
        i = int(oracle_id)
        rep = float(np.clip(base_rep[i], 0.0, 1.0))
        truth = float(self.truth_score()[i])
        recent_fail = float(np.clip(recent_fail_rate[i], 0.0, 1.0))
        cooldown = float(self.cooldown_fraction()[i])
        if self.last_step[i] >= 0:
            since_last = float(global_step - self.last_step[i])
        else:
            since_last = float(global_step)
        staleness = float(np.clip(since_last / max(0.20 * max(global_step, 1), 1.0), 0.0, 1.0))
        return float(np.clip(0.25 * (1.0 - rep) + 0.25 * (1.0 - truth) + 0.25 * recent_fail + 0.15 * cooldown + 0.10 * staleness, 0.0, 1.0))

    def audit_probability(self, oracle_id: int, base_rep: np.ndarray, recent_fail_rate: np.ndarray, global_step: int) -> Tuple[float, float]:
        risk = self.audit_risk(oracle_id, base_rep, recent_fail_rate, global_step)
        p = self.cfg.base_rate + self.cfg.risk_rate * risk
        return float(np.clip(p, 0.0, self.cfg.max_rate)), risk

    def maybe_audit(
        self,
        oracle_id: int,
        audit_pass: bool,
        severity: float,
        base_rep: np.ndarray,
        recent_fail_rate: np.ndarray,
        global_step: int,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        p, risk = self.audit_probability(oracle_id, base_rep, recent_fail_rate, global_step)
        triggered = bool(rng.random() <= p)
        if triggered:
            self.update(oracle_id, audit_pass, severity, base_rep, global_step)
        return {"triggered": triggered, "passed": bool(audit_pass), "probability": p, "risk": risk, "severity": float(severity)}

    def update(self, oracle_id: int, audit_pass: bool, severity: float, base_rep: np.ndarray, global_step: int) -> None:
        i = int(oracle_id)
        self.last_step[i] = float(global_step)
        self.trigger_count[i] += 1.0
        if audit_pass:
            self.alpha[i] += 1.0
            self.clean_streak[i] += 1.0
            self.pass_count[i] += 1.0
            self.cooldown[i] = max(0.0, self.cooldown[i] - 1.0)
            if self.clean_streak[i] >= self.cfg.min_clean_streak:
                base_rep[i] += self.cfg.pass_recovery * (1.0 - base_rep[i])
        else:
            sev = float(max(severity, 0.5))
            self.beta[i] += sev
            self.clean_streak[i] = 0.0
            self.fail_count[i] += 1.0
            base_rep[i] -= self.cfg.fail_penalty * sev
            if sev >= 1.5:
                self.cooldown[i] = float(self.cfg.cooldown_steps)
        base_rep[i] = float(np.clip(base_rep[i], 0.0, 1.0))

    def base_update(self, oracle_id: int, success: bool, severity: float, base_rep: np.ndarray) -> None:
        i = int(oracle_id)
        if success:
            base_rep[i] += self.cfg.base_success_recovery * (1.0 - base_rep[i])
        else:
            base_rep[i] -= self.cfg.base_fail_penalty * max(float(severity), 0.5)
        base_rep[i] = float(np.clip(base_rep[i], 0.0, 1.0))
