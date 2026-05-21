"""Real-trace matched environment adapter for TCO-DRL HCRL experiments.

This module is placed under experiments_real_trace so baseline experiment folders
and the original env.py are not modified.  It imports env.SchedulingEnv and
injects Chainlink/Binance real-trace dynamics as a *risk modifier* rather than
replacing oracle identity labels.

Key design in this adjusted version:
  - matched workload mode keeps the original synthetic request distribution;
  - oracle identity (trusted/normal/malicious) still comes from the simulator;
  - real trace only modulates deviation/staleness/risk/latency softly;
  - trusted oracles are not turned into malicious nodes by trace anomalies;
  - extra primary/backup/any-selected malicious/trusted metrics are exposed.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from env import SchedulingEnv as BaseSchedulingEnv


@dataclass
class RealTraceConfig:
    real_trace_path: str = os.path.join("experiments_real_trace",
                                        "../../../../../下载/tco_drl_real_trace_guarded_eval_fix_v2_shortpath/experiments_real_trace/data", "real_oracle_trace.csv")
    real_trace_split: str = "train"  # train/test/all
    real_trace_train_days: int = 20
    real_trace_auto_request_num: bool = False
    real_trace_eval_start: int = 0
    real_trace_time_order: bool = True
    # matched = keep original simulator workload and sample trace rows by service_type.
    # full_trace = use real trace rows as the request sequence.
    real_trace_workload_mode: str = "matched"
    # Soft trace-risk injection.  Defaults are intentionally weaker than the first pack
    # because real-trace anomaly is an observation-risk label, not oracle identity.
    real_trace_risk_strength: float = 0.20
    real_trace_feature_blend: float = 0.15
    real_trace_metric_blend: float = 0.20
    real_trace_validation_floor: float = 0.05
    real_trace_validation_ceil: float = 0.99
    real_trace_staleness_scale: float = 7200.0
    real_trace_latency_max_penalty: float = 0.25
    real_trace_deviation_scale: float = 0.01
    real_trace_success_bonus: float = 0.04
    real_trace_behavior_strength: float = 0.15
    real_trace_trusted_risk_scale: float = 0.25
    real_trace_normal_risk_scale: float = 0.55
    real_trace_malicious_risk_scale: float = 1.00
    real_trace_random_shift: bool = False
    # HCRL test-time mode selection. Greedy mode selection can collapse to one mode
    # under train/test trace shift; guarded keeps greedy primary/backup but protects
    # the mode head from 100% parallel collapse.
    hcrl_eval_mode_policy: str = "guarded"  # guarded/greedy/softmax
    hcrl_eval_mode_temperature: float = 1.25
    hcrl_eval_parallel_max_rate: float = 0.75
    hcrl_eval_min_requests_for_guard: int = 100
    hcrl_eval_q_margin: float = 0.05
    real_trace_verbose: bool = True


def strip_real_trace_args(argv: List[str]) -> RealTraceConfig:
    """Parse real-trace-only args and remove them before root utils.get_args runs."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--Real_Trace_Path", type=str, default=RealTraceConfig.real_trace_path)
    parser.add_argument("--Real_Trace_Split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--Real_Trace_Train_Days", type=int, default=20)
    parser.add_argument("--Real_Trace_Auto_Request_Num", action="store_true", default=False)
    parser.add_argument("--No_Real_Trace_Auto_Request_Num", action="store_true", default=False)
    parser.add_argument("--Real_Trace_Eval_Start", type=int, default=0)
    parser.add_argument("--Real_Trace_Workload_Mode", choices=["matched", "full_trace"], default="matched")
    parser.add_argument("--Real_Trace_Risk_Strength", type=float, default=RealTraceConfig.real_trace_risk_strength)
    parser.add_argument("--Real_Trace_Feature_Blend", type=float, default=RealTraceConfig.real_trace_feature_blend)
    parser.add_argument("--Real_Trace_Metric_Blend", type=float, default=RealTraceConfig.real_trace_metric_blend)
    parser.add_argument("--Real_Trace_Validation_Floor", type=float, default=RealTraceConfig.real_trace_validation_floor)
    parser.add_argument("--Real_Trace_Validation_Ceil", type=float, default=RealTraceConfig.real_trace_validation_ceil)
    parser.add_argument("--Real_Trace_Staleness_Scale", type=float, default=RealTraceConfig.real_trace_staleness_scale)
    parser.add_argument("--Real_Trace_Latency_Max_Penalty", type=float, default=RealTraceConfig.real_trace_latency_max_penalty)
    parser.add_argument("--Real_Trace_Deviation_Scale", type=float, default=RealTraceConfig.real_trace_deviation_scale)
    parser.add_argument("--Real_Trace_Success_Bonus", type=float, default=RealTraceConfig.real_trace_success_bonus)
    parser.add_argument("--Real_Trace_Behavior_Strength", type=float, default=RealTraceConfig.real_trace_behavior_strength)
    parser.add_argument("--Real_Trace_Trusted_Risk_Scale", type=float, default=RealTraceConfig.real_trace_trusted_risk_scale)
    parser.add_argument("--Real_Trace_Normal_Risk_Scale", type=float, default=RealTraceConfig.real_trace_normal_risk_scale)
    parser.add_argument("--Real_Trace_Malicious_Risk_Scale", type=float, default=RealTraceConfig.real_trace_malicious_risk_scale)
    parser.add_argument("--Real_Trace_Random_Shift", action="store_true", default=False)
    parser.add_argument("--HCRL_Eval_Mode_Policy", choices=["guarded", "greedy", "softmax"], default=RealTraceConfig.hcrl_eval_mode_policy)
    parser.add_argument("--HCRL_Eval_Mode_Temperature", type=float, default=RealTraceConfig.hcrl_eval_mode_temperature)
    parser.add_argument("--HCRL_Eval_Parallel_Max_Rate", type=float, default=RealTraceConfig.hcrl_eval_parallel_max_rate)
    parser.add_argument("--HCRL_Eval_Min_Requests_For_Guard", type=int, default=RealTraceConfig.hcrl_eval_min_requests_for_guard)
    parser.add_argument("--HCRL_Eval_Q_Margin", type=float, default=RealTraceConfig.hcrl_eval_q_margin)
    parser.add_argument("--Real_Trace_No_Verbose", action="store_true", default=False)

    ns, rest = parser.parse_known_args(argv[1:])
    argv[:] = [argv[0]] + rest
    auto = bool(ns.Real_Trace_Auto_Request_Num) and not bool(ns.No_Real_Trace_Auto_Request_Num)
    return RealTraceConfig(
        real_trace_path=ns.Real_Trace_Path,
        real_trace_split=ns.Real_Trace_Split,
        real_trace_train_days=int(ns.Real_Trace_Train_Days),
        real_trace_auto_request_num=auto,
        real_trace_eval_start=int(ns.Real_Trace_Eval_Start),
        real_trace_workload_mode=str(ns.Real_Trace_Workload_Mode),
        real_trace_risk_strength=float(ns.Real_Trace_Risk_Strength),
        real_trace_feature_blend=float(ns.Real_Trace_Feature_Blend),
        real_trace_metric_blend=float(ns.Real_Trace_Metric_Blend),
        real_trace_validation_floor=float(ns.Real_Trace_Validation_Floor),
        real_trace_validation_ceil=float(ns.Real_Trace_Validation_Ceil),
        real_trace_staleness_scale=float(ns.Real_Trace_Staleness_Scale),
        real_trace_latency_max_penalty=float(ns.Real_Trace_Latency_Max_Penalty),
        real_trace_deviation_scale=float(ns.Real_Trace_Deviation_Scale),
        real_trace_success_bonus=float(ns.Real_Trace_Success_Bonus),
        real_trace_behavior_strength=float(ns.Real_Trace_Behavior_Strength),
        real_trace_trusted_risk_scale=float(ns.Real_Trace_Trusted_Risk_Scale),
        real_trace_normal_risk_scale=float(ns.Real_Trace_Normal_Risk_Scale),
        real_trace_malicious_risk_scale=float(ns.Real_Trace_Malicious_Risk_Scale),
        real_trace_random_shift=bool(ns.Real_Trace_Random_Shift),
        hcrl_eval_mode_policy=str(ns.HCRL_Eval_Mode_Policy),
        hcrl_eval_mode_temperature=float(ns.HCRL_Eval_Mode_Temperature),
        hcrl_eval_parallel_max_rate=float(ns.HCRL_Eval_Parallel_Max_Rate),
        hcrl_eval_min_requests_for_guard=int(ns.HCRL_Eval_Min_Requests_For_Guard),
        hcrl_eval_q_margin=float(ns.HCRL_Eval_Q_Margin),
        real_trace_verbose=not bool(ns.Real_Trace_No_Verbose),
    )


def _resolve_trace_path(path: str) -> str:
    p = os.path.expanduser(path)
    if os.path.isabs(p):
        return p
    cwd_path = os.path.abspath(p)
    if os.path.exists(cwd_path):
        return cwd_path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "../../../../../下载/tco_drl_real_trace_guarded_eval_fix_v2_shortpath"))
    return os.path.join(repo_root, p)


def load_and_split_trace(path: str, split: str = "train", train_days: int = 20) -> pd.DataFrame:
    path = _resolve_trace_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Real trace CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"timestamp", "asset", "oracle_price", "reference_price", "deviation", "staleness", "validation_success", "anomaly_label", "service_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Real trace CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "service_type"])
    df["service_type"] = df["service_type"].astype(int)
    df["deviation"] = pd.to_numeric(df["deviation"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["staleness"] = pd.to_numeric(df["staleness"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["latency"] = pd.to_numeric(df.get("latency", df["staleness"]), errors="coerce").fillna(df["staleness"])
    df["validation_success"] = pd.to_numeric(df["validation_success"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["anomaly_label"] = df["anomaly_label"].fillna("suspicious").astype(str).str.lower()
    df = df.sort_values(["timestamp", "service_type", "asset"]).reset_index(drop=True)

    if split != "all":
        t0 = df["timestamp"].min()
        cutoff = t0 + pd.Timedelta(days=int(train_days))
        if split == "train":
            df = df[df["timestamp"] < cutoff]
        elif split == "test":
            df = df[df["timestamp"] >= cutoff]
        else:
            raise ValueError(f"Unknown split: {split}")
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Real trace split '{split}' is empty. Check path/date/train_days.")
    return df


def attach_real_trace_args(args, cfg: RealTraceConfig):
    """Attach real-trace config to parsed project args."""
    df = load_and_split_trace(cfg.real_trace_path, cfg.real_trace_split, cfg.real_trace_train_days)
    service_types = sorted(map(int, df["service_type"].dropna().unique().tolist()))
    args.Use_Real_Trace = True
    for k, v in asdict(cfg).items():
        setattr(args, "".join(part.capitalize() for part in k.split("_")), v)
    args.Real_Trace_Path = cfg.real_trace_path
    args.Real_Trace_Split = cfg.real_trace_split
    args.Real_Trace_Train_Days = cfg.real_trace_train_days
    args.Real_Trace_Auto_Request_Num = cfg.real_trace_auto_request_num
    args.Real_Trace_Eval_Start = cfg.real_trace_eval_start
    args.Real_Trace_Workload_Mode = cfg.real_trace_workload_mode
    args.Real_Trace_Risk_Strength = cfg.real_trace_risk_strength
    args.Real_Trace_Feature_Blend = cfg.real_trace_feature_blend
    args.Real_Trace_Metric_Blend = cfg.real_trace_metric_blend
    args.Real_Trace_Validation_Floor = cfg.real_trace_validation_floor
    args.Real_Trace_Validation_Ceil = cfg.real_trace_validation_ceil
    args.Real_Trace_Staleness_Scale = cfg.real_trace_staleness_scale
    args.Real_Trace_Latency_Max_Penalty = cfg.real_trace_latency_max_penalty
    args.Real_Trace_Deviation_Scale = cfg.real_trace_deviation_scale
    args.Real_Trace_Success_Bonus = cfg.real_trace_success_bonus
    args.Real_Trace_Behavior_Strength = cfg.real_trace_behavior_strength
    args.Real_Trace_Trusted_Risk_Scale = cfg.real_trace_trusted_risk_scale
    args.Real_Trace_Normal_Risk_Scale = cfg.real_trace_normal_risk_scale
    args.Real_Trace_Malicious_Risk_Scale = cfg.real_trace_malicious_risk_scale
    args.Real_Trace_Random_Shift = cfg.real_trace_random_shift
    args.HCRL_Eval_Mode_Policy = cfg.hcrl_eval_mode_policy
    args.HCRL_Eval_Mode_Temperature = cfg.hcrl_eval_mode_temperature
    args.HCRL_Eval_Parallel_Max_Rate = cfg.hcrl_eval_parallel_max_rate
    args.HCRL_Eval_Min_Requests_For_Guard = cfg.hcrl_eval_min_requests_for_guard
    args.HCRL_Eval_Q_Margin = cfg.hcrl_eval_q_margin
    args.Real_Trace_Time_Order = cfg.real_trace_time_order
    args.Real_Trace_Verbose = cfg.real_trace_verbose
    args.Real_Trace_Samples = int(len(df))
    args.Real_Trace_Time_Start = str(df["timestamp"].min())
    args.Real_Trace_Time_End = str(df["timestamp"].max())
    args.Real_Trace_Service_Types = service_types
    if cfg.real_trace_auto_request_num or cfg.real_trace_workload_mode == "full_trace":
        args.Request_Num = int(len(df))
    if getattr(args, "State_Mode", "original") == "original":
        args.State_Mode = "enhanced"
    if getattr(args, "Reward_Mode", "original") == "original":
        args.Reward_Mode = "risk_aware"
    if getattr(args, "Success_Mode", "original") == "original":
        args.Success_Mode = "validation_aware"
    if getattr(args, "Action_Mask_Mode", "none") == "none":
        args.Action_Mask_Mode = "type"
    if getattr(args, "Use_HCRL", False) and not getattr(args, "Disable_GNN_Encoder", False):
        args.Use_GNN_Encoder = True
    if cfg.real_trace_verbose:
        print(f"[RealTrace] path={_resolve_trace_path(cfg.real_trace_path)}")
        print(f"[RealTrace] split={cfg.real_trace_split}, train_days={cfg.real_trace_train_days}, rows={len(df)}, "
              f"time={df['timestamp'].min()} -> {df['timestamp'].max()}, services={service_types}")
        print(f"[RealTrace] workload_mode={cfg.real_trace_workload_mode}, Request_Num={args.Request_Num}, "
              f"state={args.State_Mode}, reward={args.Reward_Mode}, success={args.Success_Mode}, mask={args.Action_Mask_Mode}")
        print(f"[RealTrace] identity-preserving risk: strength={cfg.real_trace_risk_strength}, "
              f"feature_blend={cfg.real_trace_feature_blend}, metric_blend={cfg.real_trace_metric_blend}, "
              f"latency_max_penalty={cfg.real_trace_latency_max_penalty}")
        print(f"[HCRL eval mode] policy={cfg.hcrl_eval_mode_policy}, "
              f"temperature={cfg.hcrl_eval_mode_temperature}, "
              f"parallel_max_rate={cfg.hcrl_eval_parallel_max_rate}, "
              f"q_margin={cfg.hcrl_eval_q_margin}")
    return args


class RealTraceSchedulingEnv(BaseSchedulingEnv):
    """SchedulingEnv with identity-preserving real trace risk modifiers."""

    LABEL_RISK = {"normal": 0.0, "suspicious": 0.35, "anomalous": 0.75}

    def __init__(self, args):
        self.real_trace_df = load_and_split_trace(args.Real_Trace_Path, args.Real_Trace_Split, int(args.Real_Trace_Train_Days))
        self.real_trace_rows_by_service: Dict[int, pd.DataFrame] = {
            int(k): g.reset_index(drop=True) for k, g in self.real_trace_df.groupby("service_type", sort=True)
        }
        self.real_trace_request_rows = None
        super().__init__(args)

    def gen_workload(self, lamda):
        mode = str(getattr(self.args, "Real_Trace_Workload_Mode", "matched")).lower()
        if mode == "full_trace":
            self._gen_full_trace_workload(lamda)
            return
        # Matched mode: preserve original simulator's request arrival/type/MI/length distribution.
        BaseSchedulingEnv.gen_workload(self, lamda)
        self.real_trace_request_rows = self._select_trace_rows_for_request_types(self.request_type).reset_index(drop=True)
        if bool(getattr(self.args, "Real_Trace_Verbose", True)):
            counts = pd.Series(self.request_type).value_counts().sort_index().to_dict()
            print(f"[RealTrace-Matched] workload keeps original synthetic distribution; request_type_counts={counts}")
            print("[RealTrace-Matched] trace rows are sampled by matching service_type; oracle identity is not overwritten by anomaly_label")

    def _gen_full_trace_workload(self, lamda):
        lamda = max(float(lamda), 1e-8)
        n = int(self.requestNum)
        trace = self.real_trace_df
        if n <= 0:
            n = len(trace)
            self.requestNum = n
        if n > len(trace):
            reps = int(np.ceil(n / len(trace)))
            trace = pd.concat([trace] * reps, ignore_index=True).iloc[:n].copy()
        else:
            trace = trace.iloc[:n].copy()
        if bool(getattr(self.args, "Real_Trace_Random_Shift", False)) and len(trace) > 1:
            shift = int(np.random.RandomState(int(getattr(self.args, "Seed", 6))).randint(0, len(trace)))
            trace = pd.concat([trace.iloc[shift:], trace.iloc[:shift]], ignore_index=True)
        self.real_trace_request_rows = trace.reset_index(drop=True)
        self.requestNum = int(len(self.real_trace_request_rows))
        self.timeperiodNum = int(self.requestNum / max(self.timeperiodSize, 1)) + 2
        intervalT = stats_expon_rvs(scale=1.0 / lamda * 60.0, size=self.requestNum)
        self.arrival_Times = np.around(intervalT.cumsum(), 3)
        self.requestsMI = np.maximum(np.random.normal(self.requestMI, self.requestMI_std, self.requestNum).astype(int), 1)
        self.lengths = self.requestsMI / max(self.oracleCapacity, 1e-8)
        self.request_type = self.real_trace_request_rows["service_type"].astype(int).to_numpy()
        print("[RealTrace-Full] workload uses trace service_type sequence")
        print("intervalT mean: ", round(float(np.mean(intervalT)), 3), "  intervalT SD:", round(float(np.std(intervalT, ddof=1)), 3))
        print("last request arrivalT:", round(float(self.arrival_Times[-1]), 3))
        print("MI mean: ", round(float(np.mean(self.requestsMI)), 3), "  MI SD:", round(float(np.std(self.requestsMI, ddof=1)), 3))
        print("length mean: ", round(float(np.mean(self.lengths)), 3), "  length SD:", round(float(np.std(self.lengths, ddof=1)), 3))

    def _select_trace_rows_for_request_types(self, request_types):
        selected = []
        counters = {int(k): 0 for k in self.real_trace_rows_by_service.keys()}
        base_start = int(getattr(self.args, "Real_Trace_Eval_Start", 0))
        rng = np.random.RandomState(int(getattr(self.args, "Seed", 6)))
        shifts = {}
        for k, g in self.real_trace_rows_by_service.items():
            if bool(getattr(self.args, "Real_Trace_Random_Shift", False)) and len(g) > 1:
                shifts[int(k)] = int(rng.randint(0, len(g)))
            else:
                shifts[int(k)] = base_start % max(len(g), 1)
        fallback = self.real_trace_df.reset_index(drop=True)
        for t in np.asarray(request_types, dtype=int):
            g = self.real_trace_rows_by_service.get(int(t), fallback)
            if len(g) == 0:
                g = fallback
            idx = (counters.get(int(t), 0) + shifts.get(int(t), 0)) % len(g)
            selected.append(g.iloc[idx])
            counters[int(t)] = counters.get(int(t), 0) + 1
        return pd.DataFrame(selected)

    def _trace_row_for_request(self, request_attrs):
        rid = int(request_attrs[0])
        if self.real_trace_request_rows is None or len(self.real_trace_request_rows) == 0:
            return None
        rid = max(0, min(rid, len(self.real_trace_request_rows) - 1))
        return self.real_trace_request_rows.iloc[rid]

    def _trace_risk(self, row) -> float:
        if row is None:
            return 0.0
        dev_scale = max(float(getattr(self.args, "Real_Trace_Deviation_Scale", 0.01)), 1e-8)
        stale_scale = max(float(getattr(self.args, "Real_Trace_Staleness_Scale", 7200.0)), 1e-8)
        dev_risk = float(np.clip(float(row.get("deviation", 0.0)) / dev_scale, 0.0, 1.0))
        stale_risk = float(np.clip(float(row.get("staleness", 0.0)) / stale_scale, 0.0, 1.0))
        label_risk = float(self.LABEL_RISK.get(str(row.get("anomaly_label", "suspicious")).lower(), 0.35))
        # validation_success is a strict thresholded observation; keep it low-weight.
        val_penalty = 0.0 if int(row.get("validation_success", 0)) == 1 else 0.20
        return float(np.clip(0.40 * stale_risk + 0.30 * dev_risk + 0.20 * label_risk + 0.10 * val_penalty, 0.0, 1.0))

    def _trace_reliability(self, row) -> float:
        return float(np.clip(1.0 - self._trace_risk(row), 0.0, 1.0))

    def _oracle_identity_scale(self, action: int) -> float:
        if int(action) in set(map(int, self.trusted_oracles)):
            return float(getattr(self.args, "Real_Trace_Trusted_Risk_Scale", 0.25))
        if int(action) in set(map(int, self.malicious_oracles)):
            return float(getattr(self.args, "Real_Trace_Malicious_Risk_Scale", 1.0))
        return float(getattr(self.args, "Real_Trace_Normal_Risk_Scale", 0.55))

    def _base_oracle_features(self, request_attrs, policy_name):
        feats = super()._base_oracle_features(request_attrs, policy_name)
        row = self._trace_row_for_request(request_attrs)
        if row is None:
            return feats
        request_type = int(request_attrs[3])
        same_type = (self.oracleTypes == request_type).astype(float)
        rel = self._trace_reliability(row)
        risk = self._trace_risk(row)
        blend = float(np.clip(getattr(self.args, "Real_Trace_Feature_Blend", 0.15), 0.0, 1.0))
        feats[:, 5] = np.clip((1.0 - same_type) * feats[:, 5] + same_type * ((1.0 - blend) * feats[:, 5] + blend * rel), 0.0, 1.0)
        feats[:, 7] = np.clip((1.0 - same_type) * feats[:, 7] + same_type * ((1.0 - blend) * feats[:, 7] + blend * risk), 0.0, 1.0)
        return feats

    def _estimated_oracle_metrics(self, request_attrs, policy_name):
        duration, rep, obs, risk, ontime = super()._estimated_oracle_metrics(request_attrs, policy_name)
        row = self._trace_row_for_request(request_attrs)
        if row is None:
            return duration, rep, obs, risk, ontime
        request_type = int(request_attrs[3])
        same_type = (self.oracleTypes == request_type).astype(float)
        trace_risk = self._trace_risk(row)
        extra_delay = self._trace_delay_penalty(row)
        blend = float(np.clip(getattr(self.args, "Real_Trace_Metric_Blend", 0.20), 0.0, 1.0))
        duration = duration + same_type * extra_delay
        risk = np.clip((1.0 - same_type) * risk + same_type * ((1.0 - blend) * risk + blend * trace_risk), 0.0, 1.0)
        ontime = np.clip(1.0 - duration / max(float(request_attrs[4]) * 1.5, 1e-8), 0.0, 1.0)
        return duration, rep, obs, risk, ontime

    def _trace_delay_penalty(self, row) -> float:
        stale_scale = max(float(getattr(self.args, "Real_Trace_Staleness_Scale", 7200.0)), 1e-8)
        max_penalty = float(getattr(self.args, "Real_Trace_Latency_Max_Penalty", 0.25))
        stale_component = float(np.clip(float(row.get("staleness", 0.0)) / stale_scale, 0.0, 1.0))
        dev_component = float(np.clip(float(row.get("deviation", 0.0)) / max(float(getattr(self.args, "Real_Trace_Deviation_Scale", 0.01)), 1e-8), 0.0, 1.0))
        return float(max_penalty * (0.75 * stale_component + 0.25 * dev_component))

    def _simulate_oracle_attempt(self, request_attrs, action, policy_name, arrival_override=None):
        att = super()._simulate_oracle_attempt(request_attrs, action, policy_name, arrival_override=arrival_override)
        row = self._trace_row_for_request(request_attrs)
        if row is None:
            return att
        action = int(action)
        request_type = int(request_attrs[3])
        if int(self.oracleTypes[action]) != request_type:
            return att
        trace_risk = self._trace_risk(row)
        risk_strength = float(getattr(self.args, "Real_Trace_Risk_Strength", 0.20))
        floor = float(getattr(self.args, "Real_Trace_Validation_Floor", 0.05))
        ceil = float(getattr(self.args, "Real_Trace_Validation_Ceil", 0.99))
        base_prob = float(self._effective_validation_prob(action, policy_name))
        trace_success = int(row.get("validation_success", 0))
        identity_scale = self._oracle_identity_scale(action)
        success_bonus = float(getattr(self.args, "Real_Trace_Success_Bonus", 0.04)) * trace_success
        if att["is_malicious"] == 1:
            success_bonus *= 0.25
        # Identity-preserving combination: trace lowers reliability softly, but a trusted
        # node remains mostly trusted and a malicious node remains intrinsically risky.
        p_val = np.clip(base_prob * (1.0 - risk_strength * identity_scale * trace_risk) + success_bonus, floor, ceil)
        att["validation_raw"] = 1 if np.random.rand() < p_val else 0

        label = str(row.get("anomaly_label", "suspicious")).lower()
        behavior_strength = float(np.clip(getattr(self.args, "Real_Trace_Behavior_Strength", 0.15), 0.0, 1.0))
        if label == "anomalous":
            if att["is_malicious"] == 1:
                if np.random.rand() < behavior_strength * (0.60 + 0.40 * trace_risk):
                    att["behavior_record"] = max(float(att["behavior_record"]), 100.0)
            elif int(action) in set(map(int, self.trusted_oracles)):
                if np.random.rand() < behavior_strength * 0.08 * trace_risk:
                    att["behavior_record"] = max(float(att["behavior_record"]), 5.0)
            else:
                if np.random.rand() < behavior_strength * 0.25 * trace_risk:
                    att["behavior_record"] = max(float(att["behavior_record"]), 5.0)
        elif label == "suspicious":
            if att["is_malicious"] == 1 and np.random.rand() < behavior_strength * 0.25 * (0.5 + trace_risk):
                att["behavior_record"] = max(float(att["behavior_record"]), 5.0)
            elif att["is_malicious"] == 0 and np.random.rand() < behavior_strength * 0.05 * trace_risk:
                att["behavior_record"] = max(float(att["behavior_record"]), 5.0)

        extra_delay = self._trace_delay_penalty(row)
        att["durationT"] = float(att["durationT"] + extra_delay)
        att["leaveT"] = float(att["leaveT"] + extra_delay)
        att["trace_deviation"] = float(row.get("deviation", 0.0))
        att["trace_staleness"] = float(row.get("staleness", 0.0))
        att["trace_risk"] = float(trace_risk)
        att["trace_validation_success"] = trace_success
        att["trace_anomaly_label"] = label
        att["trace_identity_scale"] = float(identity_scale)
        return att

    def get_totalAnySelectedMaliciousRate(self, baseline_num, startP=0):
        out = []
        s = int(startP)
        for n in self.policy_names:
            prim = self.pb_records[n][6, s:]
            bmal = self.pb_records[n][7, s:]
            out.append(float(np.mean((prim + bmal) > 0)))
        return np.array(out)

    def get_totalAnySelectedTrustedRate(self, baseline_num, startP=0):
        out = []
        s = int(startP)
        for n in self.policy_names:
            prim = self.pb_records[n][8, s:]
            btru = self.pb_records[n][9, s:]
            out.append(float(np.mean((prim + btru) > 0)))
        return np.array(out)


# scipy.stats is imported in root env.py but not exported; keep this adapter local.
def stats_expon_rvs(scale, size):
    from scipy import stats
    return stats.expon.rvs(scale=scale, size=size)
