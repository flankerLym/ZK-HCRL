"""Trace exporters for HCRL/Audit/ZK-VOS experiments.

Drop this file next to main.py in `TCO-DRL_with baseline/`.
It records per-request HCRL scheduling evidence during normal training/evaluation,
so the generated CSV/JSONL files can be used for batch ZK-VOS proof generation,
mutation tests, DeFi trace-driven analysis, and closed-loop audit case studies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


class HCRLTraceExporter:
    """Collects HCRL decision, execution, audit, and oracle-pool traces.

    The exporter is intentionally lightweight and only uses public attributes or
    existing helper methods from SchedulingEnv. It does not alter the training
    policy, reward function, replay buffers, or environment transition logic.
    """

    POLICY = "HCRL-Oracle"

    def __init__(self, run_dir: Path, run_id: str, args: Any):
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.args = args
        self.enabled = bool(getattr(args, "Export_HCRL_Trace", True))
        self.export_pool = bool(getattr(args, "Export_Oracle_Pool_Snapshot", True))
        self.max_rows = int(getattr(args, "HCRL_Trace_Max_Rows", 0) or 0)
        self.zk_rows: List[Dict[str, Any]] = []
        self.execution_rows: List[Dict[str, Any]] = []
        self.audit_rows: List[Dict[str, Any]] = []
        self.pool_rows: List[Dict[str, Any]] = []
        self._pre_decision_cache: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _f(x: Any, default: float = 0.0) -> float:
        try:
            y = float(x)
            if np.isnan(y) or np.isinf(y):
                return default
            return y
        except Exception:
            return default

    @staticmethod
    def _i(x: Any, default: int = 0) -> int:
        try:
            if x is None:
                return default
            return int(x)
        except Exception:
            return default

    @staticmethod
    def _scale(x: Any, factor: float) -> int:
        return int(round(HCRLTraceExporter._f(x) * factor))

    def _should_record(self) -> bool:
        return self.enabled and (self.max_rows <= 0 or len(self.zk_rows) < self.max_rows)

    def _mode_name(self, args: Any, mode_action: int) -> str:
        names = list(getattr(args, "HCRL_Mode_Names", []))
        if 0 <= int(mode_action) < len(names):
            return str(names[int(mode_action)])
        return str(mode_action)

    def _thresholds(self, args: Any, request_attrs: Iterable[Any]) -> Dict[str, int]:
        # Defaults are deliberately conservative and can be overridden through args
        # if later added to param_parser.py. They are exported as integers to match
        # the existing ZK-VOS circuit input convention.
        reputation_threshold = self._f(getattr(args, "ZK_Reputation_Threshold", 0.60))
        cost_budget = self._f(getattr(args, "HCRL_Cost_Budget", 1.0))
        risk_budget = self._f(getattr(args, "HCRL_Risk_Budget", 0.06))
        # For ZK latency verification, use the request deadline by default. This
        # makes the exported trace directly compatible with deadline constraints.
        deadline = self._f(list(request_attrs)[4])
        return {
            "reputationThreshold": self._scale(reputation_threshold, 10000),
            "costBudget": self._scale(cost_budget, 1000),
            "riskBudget": self._scale(risk_budget, 10000),
            "deadline": self._scale(deadline, 1000),
        }

    def capture_decision(
        self,
        env: Any,
        args: Any,
        run_id: str,
        episode: int,
        request_attrs: Iterable[Any],
        mode_action: int,
        primary_action: int,
        backup_action: Optional[int],
        time_period: int,
        global_step: int,
    ) -> None:
        """Capture the HCRL decision evidence before env.feedback_hcrl()."""
        if not self._should_record():
            return

        policy = self.POLICY
        request_attrs = list(request_attrs)
        rid = self._i(request_attrs[0])
        selected = self._i(primary_action)
        backup = self._i(backup_action, -1)
        mode_id = self._i(mode_action)
        mode_name = self._mode_name(args, mode_id)

        duration, rep, obs, risk, ontime = env._estimated_oracle_metrics(request_attrs, policy)
        cooldown = env._audit_cooldown_fraction(policy)
        audit_truth = env.audit_truth_score(policy)
        thresholds = self._thresholds(args, request_attrs)

        service_type = self._i(env.oracleTypes[selected])
        request_service = self._i(request_attrs[3])
        service_match = int(service_type == request_service)

        rep_eff = self._f(rep[selected])
        cost = self._f(env.oracleCost[selected])
        risk_v = self._f(risk[selected])
        latency = self._f(duration[selected])
        cooldown_v = self._f(cooldown[selected])

        rep_scaled = self._scale(rep_eff, 10000)
        cost_scaled = self._scale(cost, 1000)
        risk_scaled = self._scale(risk_v, 10000)
        latency_scaled = self._scale(latency, 1000)
        cooldown_flag = int(cooldown_v > 0)

        zk_is_compliant = int(
            rep_scaled >= thresholds["reputationThreshold"]
            and cost_scaled <= thresholds["costBudget"]
            and risk_scaled <= thresholds["riskBudget"]
            and latency_scaled <= thresholds["deadline"]
            and cooldown_flag == 0
            and service_match == 1
        )

        row = {
            "run_id": run_id,
            "episode": self._i(episode),
            "time_period": self._i(time_period),
            "global_step": self._i(global_step),
            "request_id": rid,
            "request_service_type": request_service,
            "selectedOracleId": selected,
            "backupOracleId": backup,
            "mode_id": mode_id,
            "mode_name": mode_name,
            "oracleServiceType": service_type,
            "service_match": service_match,
            "repEff": rep_eff,
            "repEff_scaled": rep_scaled,
            "observed_success": self._f(obs[selected]),
            "cost": cost,
            "cost_scaled": cost_scaled,
            "risk": risk_v,
            "risk_scaled": risk_scaled,
            "latencyEst": latency,
            "latencyEst_scaled": latency_scaled,
            "ontime_prob": self._f(ontime[selected]),
            "audit_truth": self._f(audit_truth[selected]),
            "cooldown": cooldown_v,
            "cooldown_flag": cooldown_flag,
            **thresholds,
            "zk_is_compliant": zk_is_compliant,
            "is_malicious": int(selected in list(getattr(env, "malicious_oracles", []))),
            "is_trusted": int(selected in list(getattr(env, "trusted_oracles", []))),
        }
        self.zk_rows.append(row)
        self._pre_decision_cache[rid] = row

        if self.export_pool:
            oracles = []
            for oid in range(int(env.oracleNum)):
                oracles.append({
                    "oracleId": int(oid),
                    "oracleServiceType": int(env.oracleTypes[oid]),
                    "repEff": self._scale(rep[oid], 10000),
                    "cost": self._scale(env.oracleCost[oid], 1000),
                    "risk": self._scale(risk[oid], 10000),
                    "latencyEst": self._scale(duration[oid], 1000),
                    "cooldown": int(self._f(cooldown[oid]) > 0),
                    "auditTruth": self._scale(audit_truth[oid], 10000),
                })
            self.pool_rows.append({
                "run_id": run_id,
                "episode": self._i(episode),
                "request_id": rid,
                "selectedOracleId": selected,
                "oracles": oracles,
            })

    def capture_execution(
        self,
        env: Any,
        args: Any,
        run_id: str,
        episode: int,
        request_attrs: Iterable[Any],
        mode_action: int,
        primary_action: int,
        backup_action: Optional[int],
        feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture execution/audit outcomes after env.feedback_hcrl()."""
        if not self.enabled:
            return

        policy = self.POLICY
        request_attrs = list(request_attrs)
        rid = self._i(request_attrs[0])
        mode_id = self._i(mode_action)
        selected = self._i(primary_action)
        backup = self._i(backup_action, -1)

        events = env.events[policy]
        pb = env.pb_records[policy]
        audit = env.audit_records[policy]
        rep_after = env._effective_reputation_vector(policy)
        cooldown_after = env._audit_cooldown_fraction(policy)
        audit_truth_after = env.audit_truth_score(policy)

        self.execution_rows.append({
            "run_id": run_id,
            "episode": self._i(episode),
            "request_id": rid,
            "selectedOracleId": selected,
            "backupOracleId": backup,
            "mode_id": mode_id,
            "mode_name": self._mode_name(args, mode_id),
            "success": self._i(events[7, rid]),
            "success_in_time": self._i(events[10, rid]),
            "primary_success": self._i(pb[0, rid]),
            "backup_used": self._i(pb[1, rid]),
            "backup_success": self._i(pb[2, rid]),
            "backup_recovery": self._i(pb[3, rid]),
            "backup_skipped": self._i(pb[10, rid]),
            "final_duration": self._f(events[3, rid]),
            "final_leave_time": self._f(events[4, rid]),
            "total_cost": self._f(events[8, rid]),
            "match": self._i(events[9, rid]),
            "reward": self._f(events[5, rid]),
            "primary_malicious": self._i(pb[6, rid]),
            "backup_malicious": self._i(pb[7, rid]),
            "primary_trusted": self._i(pb[8, rid]),
            "backup_trusted": self._i(pb[9, rid]),
            "cost_violation": self._i(pb[17, rid]),
            "latency_violation": self._i(pb[18, rid]),
            "risk_violation": self._i(pb[19, rid]),
            "any_violation": self._i(pb[16, rid]),
            "mode_reward": self._f((feedback or {}).get("mode_reward", 0.0)),
            "primary_reward": self._f((feedback or {}).get("primary_reward", 0.0)),
            "backup_reward": self._f((feedback or {}).get("backup_reward", 0.0)),
            "repEff_after": self._f(rep_after[selected]),
            "audit_truth_after": self._f(audit_truth_after[selected]),
            "cooldown_after": self._f(cooldown_after[selected]),
        })

        pre = self._pre_decision_cache.get(rid, {})
        self.audit_rows.append({
            "run_id": run_id,
            "episode": self._i(episode),
            "request_id": rid,
            "oracle_id": selected,
            "audit_trigger": self._i(audit[0, rid]),
            "audit_pass": self._i(audit[1, rid]),
            "audit_fail": self._i(audit[2, rid]),
            "audit_truth_before": self._f(pre.get("audit_truth", 0.0)),
            "audit_truth_after": self._f(audit_truth_after[selected]),
            "repEff_before": self._f(pre.get("repEff", 0.0)),
            "repEff_after": self._f(rep_after[selected]),
            "cooldown_before": self._f(pre.get("cooldown", 0.0)),
            "cooldown_after": self._f(cooldown_after[selected]),
        })

    def save(self) -> Dict[str, str]:
        """Write all trace files under RUN_DIR and return path manifest."""
        if not self.enabled:
            return {}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, str] = {}

        def _save_csv(name: str, rows: List[Dict[str, Any]]) -> None:
            path = self.run_dir / f"{self.run_id}_{name}.csv"
            if rows:
                pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            else:
                pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
            manifest[name] = str(path)
            print(f"Saved HCRL trace {name}: {path}")

        _save_csv("hcrl_zk_schedule_trace", self.zk_rows)
        _save_csv("hcrl_execution_trace", self.execution_rows)
        _save_csv("hcrl_audit_trace", self.audit_rows)

        if self.export_pool:
            pool_path = self.run_dir / f"{self.run_id}_oracle_pool_snapshot.jsonl"
            with open(pool_path, "w", encoding="utf-8") as f:
                for row in self.pool_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest["oracle_pool_snapshot"] = str(pool_path)
            print(f"Saved HCRL trace oracle_pool_snapshot: {pool_path}")

        manifest_path = self.run_dir / f"{self.run_id}_trace_manifest.json"
        summary = {
            "run_id": self.run_id,
            "num_zk_schedule_rows": len(self.zk_rows),
            "num_execution_rows": len(self.execution_rows),
            "num_audit_rows": len(self.audit_rows),
            "num_pool_snapshots": len(self.pool_rows),
            "files": manifest,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        manifest["trace_manifest"] = str(manifest_path)
        print(f"Saved HCRL trace manifest: {manifest_path}")
        return manifest
