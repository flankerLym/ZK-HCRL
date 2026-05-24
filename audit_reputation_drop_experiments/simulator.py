"""Standalone simulator focused on whether audit lowers malicious reputation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from .audit_core import AuditConfig, AuditReputation
    from .attack_scenarios import AttackScenario, attack_intensity, attack_markers, phase_label
except ImportError:
    from audit_core import AuditConfig, AuditReputation
    from attack_scenarios import AttackScenario, attack_intensity, attack_markers, phase_label


REQUIRED_TRACE_COLUMNS = {
    "timestamp", "asset", "oracle_price", "reference_price", "deviation",
    "staleness", "validation_success", "anomaly_label", "service_type"
}


class TraceProvider:
    def __init__(self, trace_path: str | Path, split: str = "all", train_days: int = 20):
        self.path = Path(trace_path)
        if not self.path.exists():
            raise FileNotFoundError("Real trace CSV not found: %s" % self.path)
        df = pd.read_csv(self.path)
        missing = REQUIRED_TRACE_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError("Trace CSV missing required columns: %s" % sorted(missing))
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "service_type"]).sort_values("timestamp").reset_index(drop=True)
        df["service_type"] = pd.to_numeric(df["service_type"], errors="coerce").fillna(0).astype(int)
        df["deviation"] = pd.to_numeric(df["deviation"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["staleness"] = pd.to_numeric(df["staleness"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["validation_success"] = pd.to_numeric(df["validation_success"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        df["anomaly_label"] = df["anomaly_label"].fillna("suspicious").astype(str).str.lower()
        if split != "all":
            t0 = df["timestamp"].min()
            cutoff = t0 + pd.Timedelta(days=int(train_days))
            if split == "train":
                df = df[df["timestamp"] < cutoff]
            elif split == "test":
                df = df[df["timestamp"] >= cutoff]
            else:
                raise ValueError("split must be train/test/all")
        if df.empty:
            raise ValueError("Trace split '%s' is empty" % split)
        self.df = df.reset_index(drop=True)
        self.by_service = {int(k): g.reset_index(drop=True) for k, g in self.df.groupby("service_type", sort=True)}
        self.services = sorted(self.by_service.keys())
        self._cursor = {k: 0 for k in self.services}

    @staticmethod
    def trace_risk(row: pd.Series, deviation_scale: float = 0.01, staleness_scale: float = 7200.0) -> float:
        label = str(row.get("anomaly_label", "suspicious")).lower()
        label_risk = {"normal": 0.0, "suspicious": 0.35, "anomalous": 0.75}.get(label, 0.35)
        deviation = float(row.get("deviation", 0.0))
        staleness = float(row.get("staleness", 0.0))
        validation_fail = 1.0 - float(row.get("validation_success", 1.0))
        return float(np.clip(0.35 * label_risk + 0.30 * np.clip(deviation / deviation_scale, 0.0, 1.0) +
                             0.20 * np.clip(staleness / staleness_scale, 0.0, 1.0) + 0.15 * validation_fail, 0.0, 1.0))

    def sample(self, service_type: int, rng: np.random.Generator) -> pd.Series:
        st = int(service_type)
        if st not in self.by_service:
            st = int(rng.choice(self.services))
        g = self.by_service[st]
        idx = self._cursor[st] % len(g)
        self._cursor[st] += 1
        return g.iloc[idx]


@dataclass
class OraclePool:
    service_types: np.ndarray
    cost: np.ndarray
    token: np.ndarray
    capacity: np.ndarray
    malicious_ids: Set[int]
    trusted_ids: Set[int]
    normal_ids: Set[int]

    @classmethod
    def create(cls, n_oracles: int, n_services: int, malicious_ratio: float, trusted_ratio: float, rng: np.random.Generator) -> "OraclePool":
        service_types = np.arange(n_oracles) % max(n_services, 1)
        rng.shuffle(service_types)
        ids = np.arange(n_oracles)
        rng.shuffle(ids)
        n_mal = int(round(n_oracles * malicious_ratio))
        n_trusted = int(round(n_oracles * trusted_ratio))
        malicious = set(map(int, ids[:n_mal]))
        trusted = set(map(int, ids[n_mal:n_mal + n_trusted]))
        normal = set(map(int, ids[n_mal + n_trusted:]))
        cost = rng.uniform(0.78, 1.18, size=n_oracles)
        token = rng.uniform(0.35, 1.00, size=n_oracles)
        capacity = rng.uniform(0.85, 1.15, size=n_oracles)
        for i in malicious:
            cost[i] *= rng.uniform(0.45, 0.70)  # bait: malicious can look attractive before audit catches them
            token[i] = rng.uniform(0.88, 1.00)
            capacity[i] = rng.uniform(1.02, 1.22)
        for i in trusted:
            token[i] = rng.uniform(0.75, 1.00)
            capacity[i] = rng.uniform(0.92, 1.15)
        return cls(service_types=service_types, cost=cost, token=token, capacity=capacity,
                   malicious_ids=malicious, trusted_ids=trusted, normal_ids=normal)


def severity_from_obs(success: bool, validation_success: bool, timeout: bool, behavior_score: float, is_malicious: bool) -> float:
    sev = 0.0
    if timeout:
        sev += 0.5
    if not validation_success:
        sev += 1.0
    if behavior_score >= 0.50:
        sev += 0.5
    if behavior_score >= 0.90:
        sev += 1.0
    if is_malicious:
        sev += 0.5
    if success and sev == 0.0:
        return 0.0
    return float(np.clip(max(sev, 0.5), 0.5, 2.5))


class ReputationDropSimulator:
    def __init__(self, trace: TraceProvider, scenario: AttackScenario, seed: int, n_oracles: int = 120,
                 n_services: int = 3, malicious_ratio: float = 0.30, trusted_ratio: float = 0.50,
                 request_num: int = 6000, interval: int = 100, audit_cfg: Optional[AuditConfig] = None):
        self.trace = trace
        self.scenario = scenario
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.n_oracles = int(n_oracles)
        self.n_services = int(n_services)
        self.request_num = int(request_num)
        self.interval = int(max(interval, 1))
        self.pool = OraclePool.create(self.n_oracles, self.n_services, malicious_ratio, trusted_ratio, self.rng)
        self.audit = AuditReputation(self.n_oracles, audit_cfg or AuditConfig())
        self.base_rep = np.full(self.n_oracles, 0.55, dtype=float)
        for i in self.pool.malicious_ids:
            self.base_rep[i] = self.rng.uniform(0.80, 0.94)  # high initial reputation to make reputation drop visible
        for i in self.pool.trusted_ids:
            self.base_rep[i] = self.rng.uniform(0.66, 0.80)
        for i in self.pool.normal_ids:
            self.base_rep[i] = self.rng.uniform(0.48, 0.66)
        self.success_count = np.zeros(self.n_oracles, dtype=float)
        self.fail_count = np.zeros(self.n_oracles, dtype=float)
        self.select_count = np.zeros(self.n_oracles, dtype=float)
        self.colluder_ids = self._choose_colluders()
        self.rep_rows: List[Dict[str, Any]] = []
        self.timeline_rows: List[Dict[str, Any]] = []

    def _choose_colluders(self) -> Set[int]:
        mal = np.array(sorted(self.pool.malicious_ids), dtype=int)
        if len(mal) == 0:
            return set()
        k = max(1, int(round(len(mal) * self.scenario.collusion_fraction)))
        return set(map(int, self.rng.choice(mal, size=min(k, len(mal)), replace=False)))

    def _recent_fail_rate(self) -> np.ndarray:
        return self.fail_count / np.maximum(self.success_count + self.fail_count, 1.0)

    def _effective_rep(self) -> np.ndarray:
        return self.audit.effective_reputation(self.base_rep)

    def _select_oracle(self, service_type: int, trace_risk: float) -> int:
        candidates = np.where(self.pool.service_types == int(service_type))[0]
        if candidates.size == 0:
            candidates = np.arange(self.n_oracles)
        eff = self._effective_rep()
        recent_fail = self._recent_fail_rate()
        observed_success = (self.success_count + 2.0 * eff) / np.maximum(self.select_count + 2.0, 1.0)
        cost_norm = self.pool.cost / max(float(np.max(self.pool.cost)), 1e-8)
        token_norm = self.pool.token / max(float(np.max(self.pool.token)), 1e-8)
        audit_risk = 1.0 - self.audit.truth_score()
        score = 0.34 * eff + 0.23 * observed_success + 0.22 * token_norm - 0.28 * cost_norm - 0.18 * recent_fail - 0.16 * audit_risk
        score -= 0.08 * float(trace_risk) * audit_risk
        return int(candidates[np.argmax(score[candidates])])

    def _simulate_attempt(self, oracle_id: int, step: int, row: pd.Series, trace_risk: float) -> Dict[str, Any]:
        oid = int(oracle_id)
        is_mal = oid in self.pool.malicious_ids
        is_trusted = oid in self.pool.trusted_ids
        intensity = attack_intensity(self.scenario, step, self.request_num, is_mal, oid, self.colluder_ids, trace_risk)
        real_val = float(row.get("validation_success", 1.0))
        if is_trusted:
            success_prob = 0.93 - 0.07 * trace_risk
        elif is_mal:
            success_prob = 0.88 - 0.76 * intensity - 0.08 * (1.0 - real_val)
        else:
            success_prob = 0.84 - 0.20 * trace_risk
        success_prob = float(np.clip(success_prob, 0.02, 0.99))
        validation_success = bool(self.rng.random() < success_prob)
        duration = float((1.0 / max(self.pool.capacity[oid], 1e-8)) * (1.0 + self.rng.gamma(2.0, 0.13) + 0.35 * trace_risk + 0.65 * intensity))
        timeout = bool(duration > 1.65)
        behavior_score = float(np.clip(0.15 * trace_risk + 0.85 * intensity + self.rng.normal(0.0, 0.05), 0.0, 1.0))
        success = bool(validation_success and not timeout)
        severity = severity_from_obs(success, validation_success, timeout, behavior_score, is_mal)
        return {"oracle_id": oid, "is_malicious": int(is_mal), "is_trusted": int(is_trusted), "attack_intensity": float(intensity),
                "success": int(success), "validation_success": int(validation_success), "timeout": int(timeout),
                "duration": duration, "behavior_score": behavior_score, "severity": severity}


    def _background_audit_probe(self, step: int, service_type: int, trace_row: pd.Series, trace_risk: float) -> List[Dict[str, Any]]:
        """Risk-triggered audit probes of suspicious oracles.

        This is intentionally not a policy comparison component. It isolates the
        audit module's ability to degrade reputation once an oracle exhibits
        attack behavior. During attack windows, the probe observes a small number
        of potentially suspicious malicious/colluding oracles and applies the same
        Beta-posterior audit update used for selected oracles.
        """
        phase = phase_label(self.scenario, step, self.request_num)
        if phase != "attack" and self.scenario.name != "static_malicious":
            return []
        # Audit budget: sparse probes, comparable to periodic monitoring.
        if self.rng.random() > 0.35:
            return []
        same_service_mal = [i for i in self.pool.malicious_ids if int(self.pool.service_types[i]) == int(service_type)]
        if not same_service_mal:
            same_service_mal = list(self.pool.malicious_ids)
        if not same_service_mal:
            return []
        k = 1 if self.rng.random() < 0.80 else 2
        k = min(k, len(same_service_mal))
        chosen = list(map(int, self.rng.choice(np.array(same_service_mal, dtype=int), size=k, replace=False)))
        infos: List[Dict[str, Any]] = []
        for oid in chosen:
            probe = self._simulate_attempt(oid, step, trace_row, trace_risk)
            success = bool(probe["success"])
            severity = float(probe["severity"])
            # The probe is an audit observation, not a scheduling selection; it should
            # update reputation but not select_count/success_count used by policy logs.
            self.audit.base_update(oid, success, severity, self.base_rep)
            audit_pass = bool(probe["validation_success"] == 1 and probe["timeout"] == 0 and float(probe["behavior_score"]) < 0.50)
            info = self.audit.maybe_audit(oid, audit_pass, severity, self.base_rep, self._recent_fail_rate(), step, self.rng)
            infos.append(info)
        return infos

    def _record_snapshot(self, step: int, window: pd.DataFrame) -> Dict[str, Any]:
        eff = self._effective_rep()
        base = self.base_rep
        mal = list(self.pool.malicious_ids)
        trusted = list(self.pool.trusted_ids)
        normal = list(self.pool.normal_ids)
        phase = phase_label(self.scenario, step, self.request_num)
        row = {
            "scenario": self.scenario.name,
            "seed": self.seed,
            "step": int(step),
            "phase": phase,
            "malicious_rep_mean": float(np.mean(eff[mal])) if mal else np.nan,
            "trusted_rep_mean": float(np.mean(eff[trusted])) if trusted else np.nan,
            "normal_rep_mean": float(np.mean(eff[normal])) if normal else np.nan,
            "malicious_base_rep_mean": float(np.mean(base[mal])) if mal else np.nan,
            "trusted_base_rep_mean": float(np.mean(base[trusted])) if trusted else np.nan,
            "audit_truth_malicious_mean": float(np.mean(self.audit.truth_score()[mal])) if mal else np.nan,
            "audit_truth_trusted_mean": float(np.mean(self.audit.truth_score()[trusted])) if trusted else np.nan,
            "reputation_gap": float(np.mean(eff[trusted]) - np.mean(eff[mal])) if mal and trusted else np.nan,
            "audit_rate": float(window["audit_triggered"].mean()) if not window.empty else np.nan,
            "audit_fail_rate": float(window["audit_failed"].mean()) if not window.empty else np.nan,
            "selected_malicious_rate": float(window["selected_malicious"].mean()) if not window.empty else np.nan,
            "selected_trusted_rate": float(window["selected_trusted"].mean()) if not window.empty else np.nan,
            "success_rate": float(window["success"].mean()) if not window.empty else np.nan,
            "attack_intensity_mean": float(window["attack_intensity"].mean()) if not window.empty else np.nan,
            **attack_markers(self.scenario, self.request_num),
        }
        return row

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        current = []
        for step in range(self.request_num):
            self.audit.tick()
            service_type = int(self.rng.integers(0, self.n_services))
            trace_row = self.trace.sample(service_type, self.rng)
            trace_risk = self.trace.trace_risk(trace_row)
            oid = self._select_oracle(service_type, trace_risk)
            attempt = self._simulate_attempt(oid, step, trace_row, trace_risk)
            success = bool(attempt["success"])
            severity = float(attempt["severity"])
            self.select_count[oid] += 1.0
            self.success_count[oid] += 1.0 if success else 0.0
            self.fail_count[oid] += 0.0 if success else 1.0
            self.audit.base_update(oid, success, severity, self.base_rep)
            audit_pass = bool(attempt["validation_success"] == 1 and attempt["timeout"] == 0 and float(attempt["behavior_score"]) < 0.50)
            audit_infos = [self.audit.maybe_audit(oid, audit_pass, severity, self.base_rep, self._recent_fail_rate(), step, self.rng)]
            audit_infos.extend(self._background_audit_probe(step, service_type, trace_row, trace_risk))
            rec = {
                "scenario": self.scenario.name,
                "seed": self.seed,
                "step": int(step),
                "phase": phase_label(self.scenario, step, self.request_num),
                "oracle_id": int(oid),
                "selected_malicious": int(attempt["is_malicious"]),
                "selected_trusted": int(attempt["is_trusted"]),
                "success": int(success),
                "validation_success": int(attempt["validation_success"]),
                "attack_intensity": float(attempt["attack_intensity"]),
                "severity": severity,
                "audit_triggered": int(any(bool(x["triggered"]) for x in audit_infos)),
                "audit_failed": int(any(bool(x["triggered"]) and not bool(x["passed"]) for x in audit_infos)),
                "audit_probability": float(np.mean([float(x["probability"]) for x in audit_infos])) if audit_infos else 0.0,
                "audit_risk": float(np.mean([float(x["risk"]) for x in audit_infos])) if audit_infos else 0.0,
                **attack_markers(self.scenario, self.request_num),
            }
            self.timeline_rows.append(rec)
            current.append(rec)
            if (step + 1) % self.interval == 0 or step == self.request_num - 1:
                win = pd.DataFrame(current)
                self.rep_rows.append(self._record_snapshot(step, win))
                current = []
        timeline = pd.DataFrame(self.timeline_rows)
        rep = pd.DataFrame(self.rep_rows)
        summary = self._summary(rep)
        return timeline, rep, summary

    def _summary(self, rep: pd.DataFrame) -> Dict[str, Any]:
        onset = int(round(self.scenario.onset * self.request_num))
        pre = rep[rep["step"] < onset]
        after = rep[rep["step"] >= onset]
        attack = rep[rep["phase"] == "attack"]
        if pre.empty:
            pre_ref = rep.iloc[[0]]
        else:
            pre_ref = pre.tail(max(1, min(3, len(pre))))
        after_tail = after.tail(max(1, min(5, len(after)))) if not after.empty else rep.tail(1)
        pre_mal = float(pre_ref["malicious_rep_mean"].mean())
        post_mal = float(after_tail["malicious_rep_mean"].mean())
        min_after = float(after["malicious_rep_mean"].min()) if not after.empty else float(rep["malicious_rep_mean"].min())
        pre_truth = float(pre_ref["audit_truth_malicious_mean"].mean())
        post_truth = float(after_tail["audit_truth_malicious_mean"].mean())
        pre_gap = float(pre_ref["reputation_gap"].mean())
        post_gap = float(after_tail["reputation_gap"].mean())
        drop_abs = pre_mal - post_mal
        drop_pct = 100.0 * drop_abs / max(pre_mal, 1e-8)
        truth_drop_abs = pre_truth - post_truth
        drop_lag = self._drop_lag(rep, onset, pre_mal, threshold=0.05)
        return {
            "scenario": self.scenario.name,
            "seed": self.seed,
            "requests": self.request_num,
            "interval": self.interval,
            "attack_onset_step": onset,
            "pre_attack_malicious_rep": pre_mal,
            "post_attack_malicious_rep": post_mal,
            "min_after_attack_malicious_rep": min_after,
            "malicious_rep_drop_abs": drop_abs,
            "malicious_rep_drop_pct": drop_pct,
            "pre_attack_malicious_truth": pre_truth,
            "post_attack_malicious_truth": post_truth,
            "malicious_truth_drop_abs": truth_drop_abs,
            "pre_attack_reputation_gap": pre_gap,
            "post_attack_reputation_gap": post_gap,
            "reputation_gap_increase": post_gap - pre_gap,
            "drop_lag_intervals": drop_lag,
            "attack_audit_rate": float(attack["audit_rate"].mean()) if not attack.empty else np.nan,
            "attack_audit_fail_rate": float(attack["audit_fail_rate"].mean()) if not attack.empty else np.nan,
            "attack_success_rate": float(attack["success_rate"].mean()) if not attack.empty else np.nan,
            "attack_selected_malicious_rate": float(attack["selected_malicious_rate"].mean()) if not attack.empty else np.nan,
            "audit_degradation_success": int((drop_abs >= 0.05) and ((post_gap - pre_gap) >= 0.03)),
        }

    @staticmethod
    def _drop_lag(rep: pd.DataFrame, onset: int, pre_mal: float, threshold: float = 0.05) -> int:
        after = rep[rep["step"] >= onset].reset_index(drop=True)
        target = pre_mal - threshold
        for idx, row in after.iterrows():
            if float(row["malicious_rep_mean"]) <= target:
                return int(idx)
        return -1
