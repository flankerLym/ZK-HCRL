import numpy as np
from scipy import stats


class SchedulingEnv:
    """Audit-aware oracle-selection environment.

    This is a complete drop-in replacement for the project folder.  It preserves
    the original TCO-DRL baselines and adds:
      - five HCRL modes: single_cost, single_safe, serial_safe, parallel_fast,
        parallel_safe;
      - hidden/risk-triggered audit posterior for every policy/oracle;
      - asymmetric reputation update: fast penalty, slow recovery;
      - mode masks based on deadline, backup quality, cost pressure and audit risk;
      - diagnostics for primary/backup recovery and audit behavior.
    """

    def __init__(self, args):
        self.args = args
        self.policy_names = list(args.Baselines)
        self.policy_num = len(self.policy_names)
        self.policy_name_to_id = {n: i for i, n in enumerate(self.policy_names)}
        self._load_static_settings(args)
        self._init_state_shape()
        self.arrival_Times = np.zeros(self.requestNum, dtype=float)
        self.requestsMI = np.zeros(self.requestNum, dtype=float)
        self.lengths = np.zeros(self.requestNum, dtype=float)
        self.request_type = np.zeros(self.requestNum, dtype=int)
        self._init_policy_records()
        self.gen_workload(self.lamda)

    # ------------------------------------------------------------------
    # Initialization and workload
    # ------------------------------------------------------------------
    def _load_static_settings(self, args):
        self.oracleTypes = np.asarray(args.Oracle_Type, dtype=int)
        self.oracleNum = int(args.Oracle_Num)
        if self.oracleNum != len(self.oracleTypes):
            raise ValueError("Oracle_Num must equal len(Oracle_Type)")
        self.oracleCapacity = float(args.Oracle_capacity)
        self.actionNum = self.oracleNum
        self.oracleInitialReputation = float(args.Oracle_Initial_Reputation)
        self.oracleAcc = np.asarray(args.Oracle_Acc, dtype=float)
        self.oracleCost = np.asarray(args.Oracle_Cost, dtype=float)
        self.oracleToken = np.asarray(args.Oracle_Tokens, dtype=float)
        self.oracleBehaviorProbs = np.asarray(args.Oracle_Behavior_Probs, dtype=float)
        self.oracleValidationProbs = np.asarray(args.Oracle_Validation_Probs, dtype=float)
        self.oracleFatigueSensitivity = np.asarray(getattr(args, "Oracle_Fatigue_Sensitivity", [0.0] * self.oracleNum), dtype=float)
        self.malicious_oracles = list(getattr(args, "Malicious_Oracle_Index", []))
        self.normal_oracles = list(getattr(args, "Normal_Oracle_Index", []))
        self.trusted_oracles = list(getattr(args, "Trusted_Oracle_Index", []))

        self.requestMI = float(args.Request_len_Mean)
        self.requestMI_std = float(args.Request_len_Std)
        self.requestNum = int(args.Request_Num)
        self.lamda = float(args.lamda)
        self.ddl = float(args.Request_ddl)
        self.noise_probability = float(args.Noise_Probability)
        self.noise_delay = float(args.Noise_Delay)
        self.timewindowSize = int(args.Time_Window_Size)
        self.timeperiodSize = int(args.Time_Period_Size)
        self.timeperiodNum = int(self.requestNum / max(self.timeperiodSize, 1)) + 2
        for name, arr in [("Oracle_Acc", self.oracleAcc), ("Oracle_Cost", self.oracleCost),
                          ("Oracle_Tokens", self.oracleToken), ("Oracle_Validation_Probs", self.oracleValidationProbs),
                          ("Oracle_Fatigue_Sensitivity", self.oracleFatigueSensitivity)]:
            if len(arr) != self.oracleNum:
                raise ValueError(f"{name} length must equal Oracle_Num")
        if self.oracleBehaviorProbs.shape[0] != self.oracleNum:
            raise ValueError("Oracle_Behavior_Probs length must equal Oracle_Num")

    def _init_state_shape(self):
        if self.args.State_Mode == "original":
            self.s_features = 1 + 2 * self.oracleNum
        else:
            # request type, length, deadline + 12 oracle features.
            self.s_features = 3 + 12 * self.oracleNum
        # mode state uses base state + 10 risk/audit summaries.
        self.mode_extra_features = 10

    def _init_policy_records(self):
        self.events = {}
        self.oracle_events = {}
        self.reputation_factors = {}
        self.oracle_reputation_history = {}
        self.reputation_timewindow = {}
        for name in self.policy_names:
            self.events[name] = np.zeros((11, self.requestNum), dtype=float)
            self.oracle_events[name] = np.zeros((5, self.oracleNum), dtype=float)
            self.oracle_events[name][2] = self.oracleInitialReputation
            self.reputation_factors[name] = np.zeros((4, self.oracleNum), dtype=float)
            self.oracle_reputation_history[name] = np.zeros((self.timeperiodNum, self.oracleNum), dtype=float)
            self.reputation_timewindow[name] = np.zeros((0, self.oracleNum), dtype=float)

        # 0 primary_success, 1 backup_used, 2 backup_success, 3 backup_recovery,
        # 4 primary_action, 5 backup_action, 6 primary_malicious, 7 backup_malicious,
        # 8 primary_trusted, 9 backup_trusted, 10 backup_skipped, 11 backup_score,
        # 12 mode id, 13 single-like, 14 serial-like, 15 parallel-like,
        # 16 any violation, 17 cost violation, 18 latency violation, 19 risk violation,
        # 20 lambda_cost, 21 lambda_latency, 22 lambda_risk.
        self.pb_records = {name: np.zeros((23, self.requestNum), dtype=float) for name in self.policy_names}
        for name in self.policy_names:
            self.pb_records[name][4, :] = -1
            self.pb_records[name][5, :] = -1
            self.pb_records[name][12, :] = -1
        self.backup_score_history = {name: [] for name in self.policy_names}
        self.hcrl_lambdas = {name: {"cost": float(getattr(self.args, "HCRL_Lambda_Cost", 0.70)),
                                    "latency": float(getattr(self.args, "HCRL_Lambda_Latency", 0.40)),
                                    "risk": float(getattr(self.args, "HCRL_Lambda_Risk", 1.20))}
                             for name in self.policy_names}
        self._init_audit_records()

    def _init_audit_records(self):
        a0 = float(getattr(self.args, "Audit_Alpha0", 2.0))
        b0 = float(getattr(self.args, "Audit_Beta0", 2.0))
        self.audit_alpha = {n: np.full(self.oracleNum, a0, dtype=float) for n in self.policy_names}
        self.audit_beta = {n: np.full(self.oracleNum, b0, dtype=float) for n in self.policy_names}
        self.audit_clean_streak = {n: np.zeros(self.oracleNum, dtype=float) for n in self.policy_names}
        self.audit_cooldown = {n: np.zeros(self.oracleNum, dtype=float) for n in self.policy_names}
        self.audit_last_step = {n: np.full(self.oracleNum, -1.0, dtype=float) for n in self.policy_names}
        self.audit_pass_count = {n: np.zeros(self.oracleNum, dtype=float) for n in self.policy_names}
        self.audit_fail_count = {n: np.zeros(self.oracleNum, dtype=float) for n in self.policy_names}
        # rows 0 audit_trigger, 1 audit_pass, 2 audit_fail, 3 audited_selected_oracle_truth_score_mean.
        self.audit_records = {n: np.zeros((4, self.requestNum), dtype=float) for n in self.policy_names}
        self.global_step = 0

    def reset(self, args):
        self.args = args
        self.policy_names = list(args.Baselines)
        self.policy_num = len(self.policy_names)
        self.policy_name_to_id = {n: i for i, n in enumerate(self.policy_names)}
        self._load_static_settings(args)
        self._init_state_shape()
        self.arrival_Times = np.zeros(self.requestNum, dtype=float)
        self.requestsMI = np.zeros(self.requestNum, dtype=float)
        self.lengths = np.zeros(self.requestNum, dtype=float)
        self.request_type = np.zeros(self.requestNum, dtype=int)
        self._init_policy_records()
        self.gen_workload(args.lamda)

    def gen_workload(self, lamda):
        lamda = max(float(lamda), 1e-8)
        intervalT = stats.expon.rvs(scale=1.0 / lamda * 60.0, size=self.requestNum)
        self.arrival_Times = np.around(intervalT.cumsum(), 3)
        self.requestsMI = np.maximum(np.random.normal(self.requestMI, self.requestMI_std, self.requestNum).astype(int), 1)
        self.lengths = self.requestsMI / max(self.oracleCapacity, 1e-8)
        service_type_num = int(np.max(self.oracleTypes)) + 1
        if getattr(self.args, "Scenario", "static") in ["rl_hard", "rl_harder"]:
            burstiness = float(getattr(self.args, "Burstiness", 0.80))
            types = np.zeros(self.requestNum, dtype=int)
            types[0] = np.random.randint(0, service_type_num)
            for i in range(1, self.requestNum):
                types[i] = types[i - 1] if np.random.rand() < burstiness else np.random.randint(0, service_type_num)
            self.request_type = types
        else:
            self.request_type = np.random.choice(np.arange(service_type_num), size=self.requestNum)
        print("intervalT mean: ", round(float(np.mean(intervalT)), 3), "  intervalT SD:", round(float(np.std(intervalT, ddof=1)), 3))
        print("last request arrivalT:", round(float(self.arrival_Times[-1]), 3))
        print("MI mean: ", round(float(np.mean(self.requestsMI)), 3), "  MI SD:", round(float(np.std(self.requestsMI, ddof=1)), 3))
        print("length mean: ", round(float(np.mean(self.lengths)), 3), "  length SD:", round(float(np.std(self.lengths, ddof=1)), 3))

    def workload(self, request_count):
        request_id = int(request_count) - 1
        attrs = [request_id, float(self.arrival_Times[request_id]), float(self.lengths[request_id]),
                 int(self.request_type[request_id]), float(self.ddl)]
        return request_count == self.requestNum, attrs

    # ------------------------------------------------------------------
    # Reputation, audit, state encoding
    # ------------------------------------------------------------------
    def initial_reputation(self):
        for name in self.policy_names:
            self.oracle_events[name][2] = self.oracleInitialReputation

    def reset_reputation_factors(self):
        for name in self.policy_names:
            if name != "BLOR":
                self.reputation_factors[name] = np.zeros((4, self.oracleNum), dtype=float)

    def reset_reputation_factors_BLOR(self):
        if "BLOR" in self.policy_names:
            self.reputation_factors["BLOR"] = np.zeros((4, self.oracleNum), dtype=float)

    def get_reputation_factors(self, policy_name):
        return self.reputation_factors[policy_name]

    def audit_truth_score(self, policy_name):
        return self.audit_alpha[policy_name] / np.maximum(self.audit_alpha[policy_name] + self.audit_beta[policy_name], 1e-8)

    def _audit_cooldown_fraction(self, policy_name):
        steps = max(float(getattr(self.args, "Audit_Cooldown_Steps", 300)), 1.0)
        return np.clip(self.audit_cooldown[policy_name] / steps, 0.0, 1.0)

    def _effective_reputation_vector(self, policy_name):
        base = np.clip(self.oracle_events[policy_name][2], 0.0, 1.0)
        if not getattr(self.args, "Use_Audit_Reputation", True):
            return base
        w = float(getattr(self.args, "Audit_Weight_In_Reputation", 0.30))
        truth = self.audit_truth_score(policy_name)
        cooldown_penalty = float(getattr(self.args, "Audit_Cooldown_Penalty", 0.12)) * self._audit_cooldown_fraction(policy_name)
        return np.clip((1.0 - w) * base + w * truth - cooldown_penalty, 0.0, 1.0)

    def update_reputation(self, reputation_attributes, time_period, policy_name):
        counts = reputation_attributes[0]
        val = reputation_attributes[1]
        behavior = reputation_attributes[3]
        old = self.oracle_events[policy_name][2]
        recent_success = (val + self.oracleInitialReputation * 2.0) / np.maximum(counts + 2.0, 1e-8)
        behavior_penalty = np.log1p(np.maximum(behavior / np.maximum(counts, 1.0), 0.0)) / np.log1p(100.0)
        new_rep = np.clip(0.70 * old + 0.30 * (recent_success - 0.35 * behavior_penalty), 0.0, 1.0)
        # Audit posterior participates in the effective reputation used by state/action/reward.
        if getattr(self.args, "Use_Audit_Reputation", True):
            w = float(getattr(self.args, "Audit_Weight_In_Reputation", 0.30))
            new_rep = np.clip((1.0 - w) * new_rep + w * self.audit_truth_score(policy_name)
                              - float(getattr(self.args, "Audit_Cooldown_Penalty", 0.12)) * self._audit_cooldown_fraction(policy_name), 0.0, 1.0)
        self.oracle_events[policy_name][2] = new_rep
        tp = int(min(max(time_period, 0), self.timeperiodNum - 1))
        self.oracle_reputation_history[policy_name][tp] = new_rep
        self.reputation_timewindow[policy_name] = np.vstack((self.reputation_timewindow[policy_name], new_rep[None, :]))[-self.timewindowSize:]

    def _policy_uses_gnn(self, policy_name):
        if not getattr(self.args, "Use_GNN_Encoder", False) or getattr(self.args, "Disable_GNN_Encoder", False):
            return False
        if policy_name == "HCRL-Oracle":
            return True
        return bool(getattr(self.args, "Use_GNN_For_All_RL", False) and policy_name in ["DQN", "PPO", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle"])

    def _base_oracle_features(self, request_attrs, policy_name):
        _, arrival_time, length, request_type, ddl = request_attrs
        request_type = int(request_type)
        wait = np.maximum(self.oracle_events[policy_name][0] - float(arrival_time), 0.0)
        wait_norm = np.clip(wait / max(float(ddl), 1e-8), 0.0, 3.0) / 3.0
        rep = self._effective_reputation_vector(policy_name)
        cost_norm = np.clip(self.oracleCost / max(float(np.max(self.oracleCost)), 1e-8), 0.0, 1.0)
        acc_norm = np.clip(self.oracleAcc / max(float(np.max(self.oracleAcc)), 1e-8), 0.0, 1.0)
        type_match = (self.oracleTypes == request_type).astype(float)
        counts = self.reputation_factors[policy_name][0]
        val = self.reputation_factors[policy_name][1]
        token_norm = np.clip(self.oracleToken / max(float(np.max(self.oracleToken)), 1e-8), 0.0, 1.0)
        prior = 0.5 * rep + 0.5 * token_norm
        observed_success = (val + 2.0 * prior) / np.maximum(counts + 2.0, 1e-8)
        validation_feature = np.asarray(self.oracleValidationProbs, dtype=float) if getattr(self.args, "Expose_Validation_Prob", False) else observed_success
        recent_load = np.clip(counts / max(float(self.timeperiodSize), 1.0), 0.0, 1.0)
        behavior = self.reputation_factors[policy_name][3] / np.maximum(counts, 1.0)
        behavior_risk = np.clip(np.log1p(np.maximum(behavior, 0.0)) / np.log1p(100.0), 0.0, 1.0)
        delay_est = np.clip((wait + float(length) / np.maximum(self.oracleAcc, 1e-8)) / max(float(ddl), 1e-8), 0.0, 2.0) / 2.0
        audit_truth = self.audit_truth_score(policy_name)
        audit_fail_rate = self.audit_beta[policy_name] / np.maximum(self.audit_alpha[policy_name] + self.audit_beta[policy_name], 1e-8)
        cooldown = self._audit_cooldown_fraction(policy_name)
        return np.vstack((
            wait_norm, rep, cost_norm, acc_norm, type_match, validation_feature,
            recent_load, 0.5 * behavior_risk + 0.5 * delay_est,
            token_norm, audit_truth, audit_fail_rate, cooldown,
        )).T

    def _graph_encode_oracles(self, features, request_type):
        h = np.asarray(features, dtype=float).copy()
        if h.shape[0] == 0:
            return h
        same_service = (self.oracleTypes[:, None] == self.oracleTypes[None, :]).astype(float)
        reliability = 1.0 - np.abs(h[:, 5][:, None] - h[:, 5][None, :])
        load_similarity = 1.0 - np.abs(h[:, 6][:, None] - h[:, 6][None, :])
        cost_similarity = 1.0 - np.abs(h[:, 2][:, None] - h[:, 2][None, :])
        adj = (float(getattr(self.args, "GNN_Service_Weight", 1.0)) * same_service
               + float(getattr(self.args, "GNN_Reliability_Weight", 0.45)) * reliability
               + float(getattr(self.args, "GNN_Load_Weight", 0.35)) * load_similarity
               + float(getattr(self.args, "GNN_Cost_Weight", 0.25)) * cost_similarity)
        np.fill_diagonal(adj, 0.0)
        adj = adj / np.maximum(adj.sum(axis=1, keepdims=True), 1e-8)
        self_w = float(getattr(self.args, "GNN_Self_Weight", 0.55)); neigh_w = float(getattr(self.args, "GNN_Neighbor_Weight", 0.45))
        request_gate = (self.oracleTypes == int(request_type)).astype(float)[:, None]
        for _ in range(max(int(getattr(self.args, "GNN_Message_Steps", 2)), 0)):
            h = np.tanh(self_w * h + neigh_w * adj.dot(h) + 0.05 * request_gate)
        return np.clip(0.5 * (h + 1.0), 0.0, 1.0)

    def getState(self, request_attrs, policy_name):
        _, arrival_time, length, request_type, ddl = request_attrs
        request_type = int(request_type)
        if self.args.State_Mode == "original":
            state = np.hstack(([request_type], self.oracle_events[policy_name][0] - float(arrival_time), self._effective_reputation_vector(policy_name)))
            return np.nan_to_num(state.astype(float), nan=0.0, posinf=10.0, neginf=-10.0)
        feats = self._base_oracle_features(request_attrs, policy_name)
        if self._policy_uses_gnn(policy_name):
            feats = self._graph_encode_oracles(feats, request_type)
        mean_len = float(getattr(self.args, "Request_len_Mean", 6000)) / max(float(getattr(self.args, "Oracle_capacity", 1000)), 1e-8)
        prefix = np.array([request_type / max(float(np.max(self.oracleTypes)), 1.0),
                           float(length) / max(mean_len, 1e-8),
                           float(ddl) / max(float(getattr(self.args, "Harder_Request_DDL", 6.6)), 1e-8)], dtype=float)
        state = np.hstack((prefix, feats.reshape(-1)))
        return np.nan_to_num(state.astype(float), nan=0.0, posinf=10.0, neginf=-10.0)

    def get_action_mask(self, request_attrs):
        if getattr(self.args, "Action_Mask_Mode", "none") != "type":
            return np.ones(self.oracleNum, dtype=bool)
        mask = (self.oracleTypes == int(request_attrs[3]))
        if not np.any(mask):
            mask[:] = True
        # Avoid selecting severely cooled-down oracles as normal primary when possible.
        return mask.astype(bool)

    def get_backup_action_mask(self, request_attrs, primary_action):
        mask = self.get_action_mask(request_attrs).astype(bool)
        if 0 <= int(primary_action) < self.oracleNum:
            mask[int(primary_action)] = False
        if not np.any(mask):
            mask[:] = True
            if 0 <= int(primary_action) < self.oracleNum:
                mask[int(primary_action)] = False
        if not np.any(mask):
            mask[:] = True
        return mask.astype(bool)

    def _estimated_oracle_metrics(self, request_attrs, policy_name):
        _, arrival_time, length, _, ddl = request_attrs
        wait = np.maximum(self.oracle_events[policy_name][0] - float(arrival_time), 0.0)
        exe = float(length) / np.maximum(self.oracleAcc, 1e-8)
        duration = wait + exe
        rep = self._effective_reputation_vector(policy_name)
        counts = self.reputation_factors[policy_name][0]
        val = self.reputation_factors[policy_name][1]
        observed_success = (val + 2.0 * rep) / np.maximum(counts + 2.0, 1e-8)
        behavior = self.reputation_factors[policy_name][3] / np.maximum(counts, 1.0)
        behavior_risk = np.clip(np.log1p(np.maximum(behavior, 0.0)) / np.log1p(100.0), 0.0, 1.0)
        audit_risk = 1.0 - self.audit_truth_score(policy_name)
        risk = np.clip(0.35 * (1.0 - rep) + 0.35 * (1.0 - observed_success) + 0.20 * behavior_risk + 0.10 * audit_risk, 0.0, 1.0)
        ontime_prob = np.clip(1.0 - duration / max(float(ddl) * 1.5, 1e-8), 0.0, 1.0)
        return duration, rep, observed_success, risk, ontime_prob

    def get_hcrl_mode_state(self, request_attrs, policy_name):
        base = self.getState(request_attrs, policy_name)
        primary_scores = self._primary_score_vector(request_attrs, policy_name)
        primary = int(np.argmax(primary_scores))
        backup_scores = self._backup_score_vector(request_attrs, primary, policy_name)
        valid_backup = self.get_backup_action_mask(request_attrs, primary)
        best_backup_score = float(np.max(np.where(valid_backup, backup_scores, -1e9))) if np.any(valid_backup) else -1.0
        duration, rep, obs, risk, ontime = self._estimated_oracle_metrics(request_attrs, policy_name)
        primary_risk = float(risk[primary])
        primary_ontime = float(ontime[primary])
        deadline_slack = float((request_attrs[4] - duration[primary]) / max(request_attrs[4], 1e-8))
        backup_gain = float(max(0.0, best_backup_score - primary_scores[primary]))
        backup_cost = 0.0
        if np.isfinite(best_backup_score) and np.any(valid_backup):
            b = int(np.argmax(np.where(valid_backup, backup_scores, -1e9)))
            backup_cost = float(self.oracleCost[b])
        backup_cost_pressure = float(np.clip((self.oracleCost[primary] + backup_cost) / max(float(getattr(self.args, "HCRL_Cost_Budget", 1.0)), 1e-8), 0.0, 3.0) / 3.0)
        start = max(0, int(request_attrs[0]) - 300)
        if int(request_attrs[0]) > start:
            recent_success = float(np.mean(self.events[policy_name][7, start:int(request_attrs[0])]))
            recent_risk = float(np.mean(self.pb_records[policy_name][6, start:int(request_attrs[0])] + self.pb_records[policy_name][7, start:int(request_attrs[0])])) / 2.0
            recent_audit_fail = float(np.mean(self.audit_records[policy_name][2, start:int(request_attrs[0])]))
        else:
            recent_success, recent_risk, recent_audit_fail = 0.5, 0.0, 0.0
        best_backup_audit_score = 0.0
        if np.any(valid_backup):
            best_backup_audit_score = float(np.max(self.audit_truth_score(policy_name)[valid_backup]))
        summary = np.array([
            deadline_slack, primary_risk, primary_ontime, best_backup_score, backup_gain,
            backup_cost_pressure, recent_success, recent_risk, recent_audit_fail,
            best_backup_audit_score,
        ], dtype=float)
        return np.nan_to_num(np.hstack((base, summary)), nan=0.0, posinf=10.0, neginf=-10.0)

    def _primary_score_vector(self, request_attrs, policy_name):
        duration, rep, obs, risk, ontime = self._estimated_oracle_metrics(request_attrs, policy_name)
        cost_norm = self.oracleCost / max(float(np.max(self.oracleCost)), 1e-8)
        type_match = (self.oracleTypes == int(request_attrs[3])).astype(float)
        score = 0.35 * rep + 0.25 * obs + 0.20 * ontime + 0.15 * type_match - 0.15 * cost_norm - 0.25 * risk
        return np.nan_to_num(score, nan=-1e9)

    def _backup_score_vector(self, request_attrs, primary_action, policy_name):
        duration, rep, obs, risk, ontime = self._estimated_oracle_metrics(request_attrs, policy_name)
        cost_norm = self.oracleCost / max(float(np.max(self.oracleCost)), 1e-8)
        token_norm = self.oracleToken / max(float(np.max(self.oracleToken)), 1e-8)
        score = (float(getattr(self.args, "PB_W_RECENT_SUCCESS", 0.42)) * obs
                 + float(getattr(self.args, "PB_W_REPUTATION", 0.24)) * rep
                 + float(getattr(self.args, "PB_W_TOKEN", 0.14)) * token_norm
                 + 0.18 * ontime
                 - float(getattr(self.args, "PB_W_COST", 0.10)) * cost_norm
                 - float(getattr(self.args, "PB_W_BEHAVIOR_RISK", 0.20)) * risk)
        score -= 0.08 * (self.oracleCost > float(getattr(self.args, "PB_Backup_Cost_Limit", 1.05)))
        if 0 <= int(primary_action) < self.oracleNum:
            score[int(primary_action)] = -1e9
        return np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)

    def get_hcrl_mode_mask(self, request_attrs, policy_name, primary_action=None):
        mode_names = list(getattr(self.args, "HCRL_Mode_Names", ["single_cost", "single_safe", "serial_safe", "parallel_fast", "parallel_safe"]))
        mask = np.ones(len(mode_names), dtype=bool)
        if primary_action is None:
            primary_action = int(np.argmax(self._primary_score_vector(request_attrs, policy_name)))
        backup_mask = self.get_backup_action_mask(request_attrs, primary_action)
        has_backup = bool(np.any(backup_mask))
        duration, rep, obs, risk, ontime = self._estimated_oracle_metrics(request_attrs, policy_name)
        p_risk = float(risk[int(primary_action)])
        p_duration = float(duration[int(primary_action)])
        ddl = float(request_attrs[4])
        cost_pressure = float(self.oracleCost[int(primary_action)] / max(float(getattr(self.args, "HCRL_Cost_Budget", 1.0)), 1e-8))
        best_backup_score = -1e9
        if has_backup:
            best_backup_score = float(np.max(np.where(backup_mask, self._backup_score_vector(request_attrs, primary_action, policy_name), -1e9)))
        low_truth = float(self.audit_truth_score(policy_name)[int(primary_action)]) < float(getattr(self.args, "Audit_Low_Truth_Threshold", 0.45))

        def disable(name):
            if name in mode_names:
                mask[mode_names.index(name)] = False

        if not has_backup or best_backup_score < float(getattr(self.args, "HCRL_Safety_Min_Backup_Score", 0.12)):
            for n in ["serial_safe", "parallel_fast", "parallel_safe"]:
                disable(n)
        if (ddl - p_duration) / max(ddl, 1e-8) < 0.15:
            disable("serial_safe")
        if cost_pressure > 1.05:
            disable("parallel_fast"); disable("parallel_safe")
        if p_risk > float(getattr(self.args, "HCRL_Safety_Primary_Risk_Threshold", 0.52)) or low_truth:
            disable("single_cost")
        if low_truth and has_backup:
            disable("single_safe")
        if not np.any(mask):
            # Always keep at least one safe fallback.
            if "single_safe" in mode_names:
                mask[mode_names.index("single_safe")] = True
            else:
                mask[:] = True
        return mask.astype(bool)

    # ------------------------------------------------------------------
    # Simulation, audit and rewards
    # ------------------------------------------------------------------
    def _effective_validation_prob(self, action, policy_name):
        action = int(action)
        base = float(self.oracleValidationProbs[action])
        if getattr(self.args, "Scenario", "static") in ["rl_hard", "rl_harder"]:
            recent = float(self.reputation_factors[policy_name][0, action])
            avg_recent = max(self.timeperiodSize / max(self.oracleNum, 1), 1e-8)
            overload = max(0.0, recent / avg_recent - 1.0)
            fatigue_growth = np.sqrt(overload) + 0.35 * overload if getattr(self.args, "Scenario", "static") == "rl_harder" else np.log1p(overload)
            min_prob = 0.02 if getattr(self.args, "Scenario", "static") == "rl_harder" else 0.05
            fatigue = float(getattr(self.args, "Fatigue_Strength", 1.0)) * float(self.oracleFatigueSensitivity[action]) * fatigue_growth
            base = float(np.clip(base - fatigue, min_prob, 0.99))
        if getattr(self.args, "Use_Audit_Reputation", True):
            # Audit posterior slightly calibrates the hidden validation probability.
            truth = float(self.audit_truth_score(policy_name)[action])
            base = float(np.clip(0.85 * base + 0.15 * truth, 0.01, 0.99))
        return base

    def _simulate_oracle_attempt(self, request_attrs, action, policy_name, arrival_override=None):
        _, arrival_time, length, request_type, ddl = request_attrs
        action = int(action)
        effective_arrival = float(arrival_time if arrival_override is None else arrival_override)
        acc = max(float(self.oracleAcc[action]), 1e-8)
        cost = float(self.oracleCost[action])
        oracle_type = int(self.oracleTypes[action])
        idleT = float(self.oracle_events[policy_name][0, action])
        reputation = float(self._effective_reputation_vector(policy_name)[action])
        exeT = float(length) / acc
        waitT = max(idleT - effective_arrival, 0.0)
        startT = effective_arrival + waitT
        exe_time = exeT * (1.05 if action in self.malicious_oracles else 1.0)
        if np.random.rand() < self.noise_probability:
            exe_time += self.noise_delay
        durationT = waitT + exe_time
        leaveT = startT + exe_time
        match = 1 if int(request_type) == oracle_type else 0
        validation_raw = 1 if np.random.rand() < self._effective_validation_prob(action, policy_name) else 0
        probs = np.asarray(self.oracleBehaviorProbs[action], dtype=float)
        probs = probs / max(float(probs.sum()), 1e-8)
        behavior_record = float(np.random.choice([0, 1, 5, 100], p=probs))
        return {"action": action, "startT": startT, "waitT": waitT, "exeT": exeT,
                "durationT": durationT, "leaveT": leaveT, "cost": cost, "reputation": reputation,
                "match": match, "validation_raw": validation_raw, "behavior_record": behavior_record,
                "oracle_type": oracle_type, "is_malicious": 1 if action in self.malicious_oracles else 0,
                "is_trusted": 1 if action in self.trusted_oracles else 0}

    def _is_success(self, attempt, ddl):
        if getattr(self.args, "Success_Mode", "original") == "validation_aware":
            return int(attempt["durationT"] <= ddl and attempt["match"] == 1 and attempt["validation_raw"] == 1)
        return int(attempt["durationT"] <= ddl and attempt["match"] == 1)

    def _audit_severity(self, attempt, ddl):
        severity = 0.0
        if attempt["durationT"] > ddl: severity += 0.5
        if attempt["match"] == 0: severity += 0.5
        if attempt["validation_raw"] == 0: severity += 1.0
        if attempt["behavior_record"] >= 5: severity += 0.5
        if attempt["behavior_record"] >= 100: severity += 1.0
        if attempt["is_malicious"] == 1: severity += 0.5
        return float(np.clip(severity, 0.5, 2.5))

    def compute_audit_risk(self, oracle_id, policy_name):
        i = int(oracle_id)
        rep = float(self._effective_reputation_vector(policy_name)[i])
        truth = float(self.audit_truth_score(policy_name)[i])
        cooldown = float(self._audit_cooldown_fraction(policy_name)[i])
        counts = self.reputation_factors[policy_name][0, i]
        val = self.reputation_factors[policy_name][1, i]
        recent_fail = 1.0 - float((val + 1.0) / max(counts + 2.0, 1.0))
        since_last = self.global_step - self.audit_last_step[policy_name][i] if self.audit_last_step[policy_name][i] >= 0 else self.global_step
        staleness = np.clip(since_last / max(self.requestNum * 0.20, 1.0), 0.0, 1.0)
        return float(np.clip(0.25 * (1 - rep) + 0.25 * (1 - truth) + 0.25 * recent_fail + 0.15 * cooldown + 0.10 * staleness, 0.0, 1.0))

    def maybe_trigger_audit(self, request_attrs, attempt, policy_name):
        if not getattr(self.args, "Use_Audit_Reputation", True):
            return False, True, 0.0
        request_id = int(request_attrs[0]); ddl = float(request_attrs[4]); i = int(attempt["action"])
        risk = self.compute_audit_risk(i, policy_name)
        p = float(getattr(self.args, "Audit_Base_Rate", 0.03)) + float(getattr(self.args, "Audit_Risk_Rate", 0.10)) * risk
        p = float(np.clip(p, 0.0, 0.75))
        if np.random.rand() > p:
            return False, True, 0.0
        audit_pass = bool(attempt["validation_raw"] == 1 and attempt["match"] == 1 and attempt["behavior_record"] < 5 and attempt["durationT"] <= ddl * 1.15)
        severity = 0.0 if audit_pass else self._audit_severity(attempt, ddl)
        self.update_audit_reputation(i, audit_pass, severity, policy_name)
        self.audit_records[policy_name][0, request_id] = 1.0
        self.audit_records[policy_name][1, request_id] = 1.0 if audit_pass else 0.0
        self.audit_records[policy_name][2, request_id] = 0.0 if audit_pass else 1.0
        self.audit_records[policy_name][3, request_id] = self.audit_truth_score(policy_name)[i]
        return True, audit_pass, severity

    def update_audit_reputation(self, oracle_id, audit_pass, severity, policy_name):
        i = int(oracle_id)
        self.audit_last_step[policy_name][i] = self.global_step
        if audit_pass:
            self.audit_alpha[policy_name][i] += 1.0
            self.audit_clean_streak[policy_name][i] += 1.0
            self.audit_pass_count[policy_name][i] += 1.0
            self.audit_cooldown[policy_name][i] = max(0.0, self.audit_cooldown[policy_name][i] - 1.0)
            if self.audit_clean_streak[policy_name][i] >= int(getattr(self.args, "Audit_Min_Clean_Streak", 3)):
                recover = float(getattr(self.args, "Audit_Pass_Recovery", 0.03))
                self.oracle_events[policy_name][2, i] += recover * (1.0 - self.oracle_events[policy_name][2, i])
        else:
            sev = float(max(severity, 0.5))
            self.audit_beta[policy_name][i] += sev
            self.audit_clean_streak[policy_name][i] = 0.0
            self.audit_fail_count[policy_name][i] += 1.0
            self.oracle_events[policy_name][2, i] -= float(getattr(self.args, "Audit_Fail_Penalty", 0.08)) * sev
            if sev >= 1.5:
                self.audit_cooldown[policy_name][i] = float(getattr(self.args, "Audit_Cooldown_Steps", 300))
        self.oracle_events[policy_name][2, i] = float(np.clip(self.oracle_events[policy_name][2, i], 0.0, 1.0))

    def _tick_audit_cooldowns(self):
        for name in self.policy_names:
            self.audit_cooldown[name] = np.maximum(self.audit_cooldown[name] - 1.0, 0.0)

    def _original_reward(self, exeT, durationT, cost, reputation, request_type, oracle_type):
        penalty = 0 if int(request_type) == int(oracle_type) else 1
        return float((1 + 2.5 * np.exp(1.5 - float(cost))) * (float(exeT) / max(float(durationT), 1e-8)) + float(reputation) - 4 * penalty)

    def _risk_aware_reward(self, reputation, match, successful_validation, cost, durationT, ddl, behavior_record, audit_risk=0.0):
        ddl = max(float(ddl), 1e-8)
        timeout = 1.0 if float(durationT) > ddl else 0.0
        on_time = 1.0 - timeout
        rep_score = float(np.clip(reputation, 0.0, 1.0))
        match_score = float(match); val_score = float(successful_validation)
        task_success = match_score * val_score * on_time
        cost_score = float(np.clip(float(cost), 0.0, 1.25) / 1.25)
        response_ratio = float(np.clip(float(durationT) / ddl, 0.0, 2.5))
        response_penalty = float(np.clip(min(response_ratio, 1.0) * 0.4 + max(response_ratio - 1.0, 0.0) * 0.9, 0.0, 1.0))
        behavior_risk = float(np.log1p(max(float(behavior_record), 0.0)) / np.log1p(100.0))
        a = self.args
        positive = a.W_SUCCESS * task_success + a.W_VALIDATION * val_score + a.W_MATCH * match_score + a.W_REPUTATION * rep_score
        negative = (a.W_COST * cost_score + a.W_RESPONSE * response_penalty + a.W_BEHAVIOR * behavior_risk
                    + a.W_TIMEOUT * timeout + 0.8 * (1.0 - task_success)
                    + float(getattr(a, "Audit_Risk_Reward_Penalty", 0.30)) * float(audit_risk))
        normalizer = a.W_SUCCESS + a.W_VALIDATION + a.W_MATCH + a.W_REPUTATION + a.W_COST + a.W_RESPONSE + a.W_BEHAVIOR + a.W_TIMEOUT + 0.8 + float(getattr(a, "Audit_Risk_Reward_Penalty", 0.30))
        return float(np.clip(a.Reward_Scale * (positive - negative) / max(normalizer, 1e-8), -a.Reward_Clip, a.Reward_Clip))

    def _reward_for_attempt(self, attempt, request_attrs, final_success=None, total_cost=None, final_duration=None, combined_behavior=None, combined_rep=None, audit_risk=None):
        _, _, _, request_type, ddl = request_attrs
        if audit_risk is None:
            audit_risk = 1.0 - float(self.audit_truth_score("HCRL-Oracle")[attempt["action"]]) if "HCRL-Oracle" in self.policy_names else 0.0
        if getattr(self.args, "Reward_Mode", "original") == "risk_aware":
            return self._risk_aware_reward(combined_rep if combined_rep is not None else attempt["reputation"],
                                           attempt["match"], final_success if final_success is not None else attempt["validation_raw"],
                                           total_cost if total_cost is not None else attempt["cost"],
                                           final_duration if final_duration is not None else attempt["durationT"], ddl,
                                           combined_behavior if combined_behavior is not None else attempt["behavior_record"], audit_risk=audit_risk)
        return self._original_reward(attempt["exeT"], final_duration if final_duration is not None else attempt["durationT"],
                                     total_cost if total_cost is not None else attempt["cost"], attempt["reputation"], request_type, attempt["oracle_type"])

    def _record_attempt_updates(self, policy_name, attempts):
        for att in attempts:
            if att is None: continue
            i = int(att["action"])
            self.oracle_events[policy_name][1, i] += 1
            self.oracle_events[policy_name][0, i] = max(self.oracle_events[policy_name][0, i], att["leaveT"])
            self.oracle_events[policy_name][3, i] += att["match"]
            self.oracle_events[policy_name][4, i] += att["validation_raw"]
            self.reputation_factors[policy_name][0, i] += 1
            self.reputation_factors[policy_name][1, i] += att["validation_raw"]
            self.reputation_factors[policy_name][2, i] += att["durationT"]
            self.reputation_factors[policy_name][3, i] += att["behavior_record"]

    def _record_request(self, policy_name, request_id, primary, reward, success, final_duration, final_leaveT, total_cost, match):
        self.events[policy_name][0, request_id] = primary["action"]
        self.events[policy_name][1, request_id] = primary["startT"]
        self.events[policy_name][2, request_id] = primary["waitT"]
        self.events[policy_name][3, request_id] = final_duration
        self.events[policy_name][4, request_id] = final_leaveT
        self.events[policy_name][5, request_id] = reward
        self.events[policy_name][6, request_id] = primary["exeT"]
        self.events[policy_name][7, request_id] = success
        self.events[policy_name][8, request_id] = total_cost
        self.events[policy_name][9, request_id] = match
        self.events[policy_name][10, request_id] = 1.0 if float(final_duration) <= float(self.ddl) else 0.0

    def feedback(self, request_attrs, action, policy_name):
        self.global_step += 1
        self._tick_audit_cooldowns()
        rid, _, _, _, ddl = request_attrs; rid = int(rid)
        attempt = self._simulate_oracle_attempt(request_attrs, action, policy_name)
        success = self._is_success(attempt, float(ddl))
        audit_risk = 1.0 - float(self.audit_truth_score(policy_name)[int(action)])
        reward = self._reward_for_attempt(attempt, request_attrs, final_success=success, audit_risk=audit_risk)
        self._record_attempt_updates(policy_name, [attempt])
        self.maybe_trigger_audit(request_attrs, attempt, policy_name)
        self._record_request(policy_name, rid, attempt, reward, success, attempt["durationT"], attempt["leaveT"], attempt["cost"], attempt["match"])
        self.pb_records[policy_name][0, rid] = success
        self.pb_records[policy_name][4, rid] = int(action)
        self.pb_records[policy_name][6, rid] = attempt["is_malicious"]
        self.pb_records[policy_name][8, rid] = attempt["is_trusted"]
        return reward

    # ------------------------------------------------------------------
    # Primary-backup / HCRL feedback
    # ------------------------------------------------------------------
    def choose_backup_oracle(self, request_attrs, primary_action, policy_name):
        candidates = np.where(self.get_backup_action_mask(request_attrs, primary_action))[0]
        if candidates.size == 0:
            return int(primary_action)
        random_flag = (policy_name == "COBRA-Oracle" and bool(getattr(self.args, "COBRA_Random_Backup", False))) or (policy_name == "HCRL-Oracle" and bool(getattr(self.args, "HCRL_Random_Backup", False)))
        if random_flag:
            return int(np.random.choice(candidates))
        score = self._backup_score_vector(request_attrs, primary_action, policy_name)
        return int(candidates[np.argmax(score[candidates])])

    def _should_use_backup(self, policy_name, request_attrs, primary_action, backup_score):
        if policy_name == "PB-SafeDQN":
            if getattr(self.args, "PB_Backup_Trigger", "cost_aware") == "always":
                return True
            return backup_score >= float(getattr(self.args, "PB_Min_Backup_Score", 0.38))
        if policy_name == "COBRA-Oracle":
            mode = getattr(self.args, "COBRA_Gate_Mode", "adaptive")
            if mode == "always": return True
            if mode == "never": return False
            duration, rep, obs, risk, ontime = self._estimated_oracle_metrics(request_attrs, policy_name)
            p = int(primary_action)
            threshold = float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46))
            if mode == "adaptive":
                threshold -= 0.10 * float(risk[p]) + 0.05 * (1.0 - float(ontime[p]))
            return backup_score >= threshold
        return backup_score >= float(getattr(self.args, "HCRL_Safety_Min_Backup_Score", 0.12))

    def feedback_primary_backup(self, request_attrs, primary_action, policy_name):
        self.global_step += 1
        self._tick_audit_cooldowns()
        rid, arrival, _, _, ddl = request_attrs; rid = int(rid)
        primary = self._simulate_oracle_attempt(request_attrs, primary_action, policy_name)
        primary_success = self._is_success(primary, float(ddl))
        backup_action = self.choose_backup_oracle(request_attrs, primary_action, policy_name)
        backup_score = float(self._backup_score_vector(request_attrs, primary_action, policy_name)[backup_action]) if backup_action != primary_action else -1e9
        use_backup = self._should_use_backup(policy_name, request_attrs, primary_action, backup_score)
        mode_parallel = getattr(self.args, "PB_Backup_Mode", "parallel") == "parallel" or policy_name == "COBRA-Oracle"
        backup = None
        if use_backup:
            if mode_parallel:
                backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name)
            else:
                backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name, arrival_override=primary["leaveT"])
        if backup is None:
            final_success = primary_success
            final_duration = primary["durationT"]
            final_leave = primary["leaveT"]
            total_cost = primary["cost"]
            final_match = primary["match"]
            combined_behavior = primary["behavior_record"]
            combined_rep = primary["reputation"]
        else:
            backup_success = self._is_success(backup, float(ddl))
            final_success = int(primary_success or backup_success)
            if primary_success and backup_success:
                winner = primary if primary["durationT"] <= backup["durationT"] else backup
            elif primary_success:
                winner = primary
            elif backup_success:
                winner = backup
            else:
                winner = primary if primary["durationT"] <= backup["durationT"] else backup
            final_duration = winner["durationT"] if mode_parallel else (primary["durationT"] if primary_success else backup["durationT"])
            final_leave = max(primary["leaveT"], backup["leaveT"]) if mode_parallel else (primary["leaveT"] if primary_success else backup["leaveT"])
            total_cost = primary["cost"] + backup["cost"] * (float(getattr(self.args, "HCRL_Parallel_Cost_Discount", 0.85)) if mode_parallel else 1.0)
            final_match = max(primary["match"], backup["match"])
            combined_behavior = max(primary["behavior_record"], backup["behavior_record"])
            combined_rep = max(primary["reputation"], backup["reputation"])
        audit_risk = 1.0 - float(self.audit_truth_score(policy_name)[int(primary_action)])
        if backup is not None:
            audit_risk = min(audit_risk, 1.0 - float(self.audit_truth_score(policy_name)[int(backup_action)]))
        reward = self._reward_for_attempt(primary, request_attrs, final_success=final_success, total_cost=total_cost,
                                          final_duration=final_duration, combined_behavior=combined_behavior, combined_rep=combined_rep,
                                          audit_risk=audit_risk)
        if use_backup:
            reward += float(getattr(self.args, "PB_Backup_Recovery_Bonus", 0.38)) * int((not primary_success) and final_success)
            reward -= float(getattr(self.args, "PB_Backup_Used_Penalty", 0.16))
        else:
            reward += float(getattr(self.args, "PB_Primary_Success_Bonus", 0.18)) * int(primary_success)
        reward = float(np.clip(reward, -float(getattr(self.args, "Reward_Clip", 3.0)), float(getattr(self.args, "Reward_Clip", 3.0))))
        self._record_attempt_updates(policy_name, [primary, backup])
        self.maybe_trigger_audit(request_attrs, primary, policy_name)
        if backup is not None:
            self.maybe_trigger_audit(request_attrs, backup, policy_name)
        self._record_request(policy_name, rid, primary, reward, final_success, final_duration, final_leave, total_cost, final_match)
        self._record_pb(policy_name, rid, primary, backup, primary_success, final_success, use_backup, backup_score, -1)
        return reward

    def feedback_hcrl(self, request_attrs, mode_action, primary_action, backup_action, policy_name):
        self.global_step += 1
        self._tick_audit_cooldowns()
        rid, arrival, _, _, ddl = request_attrs; rid = int(rid)
        mode_names = list(getattr(self.args, "HCRL_Mode_Names", ["single_cost", "single_safe", "serial_safe", "parallel_fast", "parallel_safe"]))
        mode_action = int(np.clip(mode_action, 0, len(mode_names) - 1))
        mode_name = mode_names[mode_action]
        primary = self._simulate_oracle_attempt(request_attrs, primary_action, policy_name)
        primary_success = self._is_success(primary, float(ddl))
        if backup_action is None or int(backup_action) < 0 or int(backup_action) == int(primary_action):
            backup_action = self.choose_backup_oracle(request_attrs, primary_action, policy_name)
        backup_score = float(self._backup_score_vector(request_attrs, primary_action, policy_name)[int(backup_action)])
        use_backup = mode_name in ["serial_safe", "parallel_fast", "parallel_safe"] and backup_score > -1e8
        parallel = mode_name in ["parallel_fast", "parallel_safe"]
        backup = None
        if use_backup:
            if parallel:
                backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name)
            else:
                backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name, arrival_override=primary["leaveT"])
        if backup is None:
            final_success = primary_success
            final_duration = primary["durationT"]
            final_leave = primary["leaveT"]
            total_cost = primary["cost"]
            final_match = primary["match"]
            combined_behavior = primary["behavior_record"]
            combined_rep = primary["reputation"]
            backup_success = 0
        else:
            backup_success = self._is_success(backup, float(ddl))
            final_success = int(primary_success or backup_success)
            if parallel:
                candidates = [a for a, s in [(primary, primary_success), (backup, backup_success)] if s]
                winner = min(candidates, key=lambda a: a["durationT"]) if candidates else min([primary, backup], key=lambda a: a["durationT"])
                final_duration = winner["durationT"]
                final_leave = max(primary["leaveT"], backup["leaveT"])
                total_cost = primary["cost"] + backup["cost"] * float(getattr(self.args, "HCRL_Parallel_Cost_Discount", 0.85))
            else:
                final_duration = primary["durationT"] if primary_success else backup["durationT"]
                final_leave = primary["leaveT"] if primary_success else backup["leaveT"]
                total_cost = primary["cost"] + (0.0 if primary_success else backup["cost"])
            final_match = max(primary["match"], backup["match"])
            combined_behavior = max(primary["behavior_record"], backup["behavior_record"])
            combined_rep = max(primary["reputation"], backup["reputation"])
        primary_audit_risk = 1.0 - float(self.audit_truth_score(policy_name)[int(primary_action)])
        backup_audit_risk = 0.0 if backup is None else 1.0 - float(self.audit_truth_score(policy_name)[int(backup_action)])
        audit_risk = max(primary_audit_risk, backup_audit_risk) if mode_name in ["parallel_safe", "serial_safe"] else primary_audit_risk
        base_reward = self._reward_for_attempt(primary, request_attrs, final_success=final_success, total_cost=total_cost,
                                               final_duration=final_duration, combined_behavior=combined_behavior, combined_rep=combined_rep,
                                               audit_risk=audit_risk)
        # Mode-specific shaping.
        mode_reward = base_reward
        primary_reward = base_reward
        backup_reward = 0.0
        cost_violation = float(total_cost > float(getattr(self.args, "HCRL_Cost_Budget", 1.0)))
        latency_violation = float(final_duration > float(getattr(self.args, "HCRL_Latency_Budget", 6.2)))
        risk_violation = float(primary_audit_risk > float(getattr(self.args, "HCRL_Risk_Budget", 0.06)))
        if mode_name == "single_cost":
            mode_reward += 0.12 * (1.0 - primary["cost"]) - 0.25 * primary_audit_risk
        elif mode_name == "single_safe":
            mode_reward += 0.18 * primary["reputation"] - 0.35 * primary_audit_risk
        elif mode_name == "serial_safe":
            mode_reward += float(getattr(self.args, "HCRL_Backup_Recovery_Bonus", 0.72)) * int((not primary_success) and final_success)
            mode_reward -= 0.10 * cost_violation + 0.12 * latency_violation
            backup_reward = mode_reward
        elif mode_name == "parallel_fast":
            mode_reward += 0.20 * int(final_duration <= ddl) - float(getattr(self.args, "HCRL_Backup_Used_Penalty", 0.08))
            mode_reward -= 0.18 * cost_violation
            backup_reward = mode_reward
        elif mode_name == "parallel_safe":
            mode_reward += 0.25 * int(final_success) - 0.30 * max(0.0, backup_audit_risk - 0.50)
            mode_reward -= 0.20 * cost_violation
            backup_reward = mode_reward
        if use_backup and primary_success:
            mode_reward -= float(getattr(self.args, "HCRL_Unnecessary_Backup_Penalty", 0.18))
        if (not use_backup) and (not primary_success):
            mode_reward -= float(getattr(self.args, "HCRL_Skip_Recovery_Penalty", 0.20))
        mode_reward = float(np.clip(mode_reward, -float(getattr(self.args, "Reward_Clip", 3.0)), float(getattr(self.args, "Reward_Clip", 3.0))))
        primary_reward = float(np.clip(primary_reward - float(getattr(self.args, "HCRL_Primary_Malicious_Penalty", 0.80)) * primary["is_malicious"],
                                       -float(getattr(self.args, "Reward_Clip", 3.0)), float(getattr(self.args, "Reward_Clip", 3.0))))
        backup_reward = float(np.clip(backup_reward - float(getattr(self.args, "HCRL_Backup_Malicious_Penalty", 1.20)) * (backup["is_malicious"] if backup else 0),
                                      -float(getattr(self.args, "Reward_Clip", 3.0)), float(getattr(self.args, "Reward_Clip", 3.0))))
        self._record_attempt_updates(policy_name, [primary, backup])
        self.maybe_trigger_audit(request_attrs, primary, policy_name)
        if backup is not None:
            self.maybe_trigger_audit(request_attrs, backup, policy_name)
        self._record_request(policy_name, rid, primary, mode_reward, final_success, final_duration, final_leave, total_cost, final_match)
        self._record_pb(policy_name, rid, primary, backup, primary_success, final_success, use_backup, backup_score, mode_action)
        self.pb_records[policy_name][16, rid] = max(cost_violation, latency_violation, risk_violation)
        self.pb_records[policy_name][17, rid] = cost_violation
        self.pb_records[policy_name][18, rid] = latency_violation
        self.pb_records[policy_name][19, rid] = risk_violation
        self.pb_records[policy_name][20, rid] = self.hcrl_lambdas[policy_name]["cost"]
        self.pb_records[policy_name][21, rid] = self.hcrl_lambdas[policy_name]["latency"]
        self.pb_records[policy_name][22, rid] = self.hcrl_lambdas[policy_name]["risk"]
        return {"mode_reward": mode_reward, "primary_reward": primary_reward, "backup_reward": backup_reward}

    def _record_pb(self, policy_name, rid, primary, backup, primary_success, final_success, use_backup, backup_score, mode_action):
        self.pb_records[policy_name][0, rid] = primary_success
        self.pb_records[policy_name][1, rid] = 1.0 if use_backup else 0.0
        self.pb_records[policy_name][2, rid] = self._is_success(backup, self.ddl) if backup is not None else 0.0
        self.pb_records[policy_name][3, rid] = 1.0 if ((not primary_success) and final_success and use_backup) else 0.0
        self.pb_records[policy_name][4, rid] = primary["action"]
        self.pb_records[policy_name][5, rid] = -1 if backup is None else backup["action"]
        self.pb_records[policy_name][6, rid] = primary["is_malicious"]
        self.pb_records[policy_name][7, rid] = 0 if backup is None else backup["is_malicious"]
        self.pb_records[policy_name][8, rid] = primary["is_trusted"]
        self.pb_records[policy_name][9, rid] = 0 if backup is None else backup["is_trusted"]
        self.pb_records[policy_name][10, rid] = 0.0 if use_backup else 1.0
        self.pb_records[policy_name][11, rid] = 0.0 if backup_score < -1e8 else backup_score
        self.pb_records[policy_name][12, rid] = mode_action
        if mode_action in [-1, 0, 1]: self.pb_records[policy_name][13, rid] = 1
        if mode_action == 2: self.pb_records[policy_name][14, rid] = 1
        if mode_action in [3, 4]: self.pb_records[policy_name][15, rid] = 1
        if backup_score > -1e8: self.backup_score_history[policy_name].append(float(backup_score))

    def feedback_PSG_FWA(self, request_attrs, policy_name):
        score = self._primary_score_vector(request_attrs, policy_name)
        if getattr(self.args, "SemiGreedy_View", "myopic") == "risk_aware":
            score -= 0.20 * (1.0 - self.audit_truth_score(policy_name))
        return score, self.oracleCost.copy()

    # ------------------------------------------------------------------
    # Getters for logging and final summaries
    # ------------------------------------------------------------------
    def _slice(self, arr, startP):
        return arr[:, int(startP):] if arr.ndim == 2 else arr[int(startP):]

    def get_oracle_idleT(self, policy_name): return self.oracle_events[policy_name][0]
    def get_request_num(self, policy_name): return self.reputation_factors[policy_name][0]
    def get_successful_validation(self, policy_name): return self.reputation_factors[policy_name][1]

    def get_totalRewards(self, baseline_num, startP=0): return np.array([np.sum(self.events[n][5, int(startP):]) for n in self.policy_names])
    def get_total_responseTs(self, baseline_num, startP=0): return np.array([np.mean(self.events[n][3, int(startP):]) for n in self.policy_names])
    def get_totalSuccess(self, baseline_num, startP=0): return np.array([np.mean(self.events[n][7, int(startP):]) for n in self.policy_names])
    def get_totalSuccessInTime(self, baseline_num, startP=0): return np.array([np.mean(self.events[n][10, int(startP):]) for n in self.policy_names])
    def get_totalTimes(self, baseline_num, startP=0): return np.array([np.mean(self.events[n][4, int(startP):]) for n in self.policy_names])
    def get_totalCost(self, baseline_num, startP=0): return np.array([np.mean(self.events[n][8, int(startP):]) for n in self.policy_names])
    def get_totalMatch(self, baseline_num, startP=0): return np.array([np.mean(self.events[n][9, int(startP):]) for n in self.policy_names])

    def get_totalAssignedMaliciousRate(self, baseline_num, startP=0): return np.array([np.mean(np.isin(self.events[n][0, int(startP):].astype(int), self.malicious_oracles)) for n in self.policy_names])
    def get_totalAssignedNormalRate(self, baseline_num, startP=0): return np.array([np.mean(np.isin(self.events[n][0, int(startP):].astype(int), self.normal_oracles)) for n in self.policy_names])
    def get_totalAssignedTrustedRate(self, baseline_num, startP=0): return np.array([np.mean(np.isin(self.events[n][0, int(startP):].astype(int), self.trusted_oracles)) for n in self.policy_names])

    def get_totalPrimarySuccessRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][0, int(startP):]) for n in self.policy_names])
    def get_totalBackupUsedRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][1, int(startP):]) for n in self.policy_names])
    def get_totalBackupSuccessRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][2, int(startP):]) for n in self.policy_names])
    def get_totalBackupRecoveryRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][3, int(startP):]) for n in self.policy_names])
    def get_totalConditionalBackupRecoveryRate(self, baseline_num, startP=0):
        out = []
        for n in self.policy_names:
            used = self.pb_records[n][1, int(startP):]
            rec = self.pb_records[n][3, int(startP):]
            out.append(float(np.sum(rec) / max(np.sum(used), 1e-8)))
        return np.array(out)
    def get_totalBackupSkippedRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][10, int(startP):]) for n in self.policy_names])
    def get_totalBackupScoreMean(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][11, int(startP):]) for n in self.policy_names])
    def get_totalPrimaryMaliciousRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][6, int(startP):]) for n in self.policy_names])
    def get_totalBackupMaliciousRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][7, int(startP):]) for n in self.policy_names])
    def get_totalPrimaryTrustedRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][8, int(startP):]) for n in self.policy_names])
    def get_totalBackupTrustedRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][9, int(startP):]) for n in self.policy_names])
    def get_totalCostPerSuccess(self, baseline_num, startP=0):
        return np.array([np.mean(self.events[n][8, int(startP):]) / max(np.mean(self.events[n][7, int(startP):]), 1e-8) for n in self.policy_names])
    def get_totalHCRLSingleModeRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][13, int(startP):]) for n in self.policy_names])
    def get_totalHCRLSerialModeRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][14, int(startP):]) for n in self.policy_names])
    def get_totalHCRLParallelModeRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][15, int(startP):]) for n in self.policy_names])
    def get_totalConstraintViolationRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][16, int(startP):]) for n in self.policy_names])
    def get_totalCostViolationRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][17, int(startP):]) for n in self.policy_names])
    def get_totalLatencyViolationRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][18, int(startP):]) for n in self.policy_names])
    def get_totalRiskViolationRate(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][19, int(startP):]) for n in self.policy_names])
    def get_totalHCRLLambdaCost(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][20, int(startP):]) for n in self.policy_names])
    def get_totalHCRLLambdaLatency(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][21, int(startP):]) for n in self.policy_names])
    def get_totalHCRLLambdaRisk(self, baseline_num, startP=0): return np.array([np.mean(self.pb_records[n][22, int(startP):]) for n in self.policy_names])
    def get_totalAuditRate(self, baseline_num, startP=0): return np.array([np.mean(self.audit_records[n][0, int(startP):]) for n in self.policy_names])
    def get_totalAuditPassRate(self, baseline_num, startP=0): return np.array([np.mean(self.audit_records[n][1, int(startP):]) for n in self.policy_names])
    def get_totalAuditFailRate(self, baseline_num, startP=0): return np.array([np.mean(self.audit_records[n][2, int(startP):]) for n in self.policy_names])
    def get_totalAuditTruthMean(self, baseline_num, startP=0): return np.array([np.mean(self.audit_truth_score(n)) for n in self.policy_names])

    # Compatibility getters used by original plotting code.
    def get_accumulateRewards(self, baseline_num, performance_c, request_c): return self.get_totalRewards(baseline_num, performance_c)
    def get_accumulateCost(self, baseline_num, performance_c, request_c): return self.get_totalCost(baseline_num, performance_c)
    def get_FinishTimes(self, baseline_num, performance_c, request_c): return self.get_totalTimes(baseline_num, performance_c)
    def get_executeTs(self, baseline_num, performance_c, request_c): return np.array([np.mean(self.events[n][6, int(performance_c):int(request_c)]) for n in self.policy_names])
    def get_waitTs(self, baseline_num, performance_c, request_c): return np.array([np.mean(self.events[n][2, int(performance_c):int(request_c)]) for n in self.policy_names])
    def get_responseTs(self, baseline_num, performance_c, request_c): return np.array([np.mean(self.events[n][3, int(performance_c):int(request_c)]) for n in self.policy_names])
    def get_successTimes(self, baseline_num, performance_c, request_c): return np.array([np.mean(self.events[n][7, int(performance_c):int(request_c)]) for n in self.policy_names])
    def get_successInTime(self, baseline_num, performance_c, request_c): return np.array([np.mean(self.events[n][10, int(performance_c):int(request_c)]) for n in self.policy_names])
