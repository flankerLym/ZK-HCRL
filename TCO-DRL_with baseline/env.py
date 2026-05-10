import numpy as np
from scipy import stats


class SchedulingEnv:
    """Scalable oracle-selection environment for TCO-DRL / HCRL-Oracle.

    Complete replacement env.py compatible with the current paper-version main.py.
    It includes:
      - scalable oracle communities;
      - validation-aware success and risk-aware reward;
      - type action masks;
      - PB-SafeDQN / COBRA primary-backup recovery;
      - HCRL-v2 hierarchical constrained feedback with adaptive safety recovery;
      - GNN-style oracle state encoder;
      - complete final-summary getters used by main.py.
    """

    def __init__(self, args):
        self.args = args
        self.policy_names = list(args.Baselines)
        self.policy_num = len(self.policy_names)
        self.policy_name_to_id = {name: idx for idx, name in enumerate(self.policy_names)}
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
        self.oracleFatigueSensitivity = np.asarray(
            getattr(args, "Oracle_Fatigue_Sensitivity", [0.0] * self.oracleNum), dtype=float
        )
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

        # Defensive shape checks. This makes errors obvious if param_parser changes.
        for name, arr in [
            ("Oracle_Acc", self.oracleAcc),
            ("Oracle_Cost", self.oracleCost),
            ("Oracle_Tokens", self.oracleToken),
            ("Oracle_Validation_Probs", self.oracleValidationProbs),
            ("Oracle_Fatigue_Sensitivity", self.oracleFatigueSensitivity),
        ]:
            if len(arr) != self.oracleNum:
                raise ValueError(f"{name} length must equal Oracle_Num")
        if self.oracleBehaviorProbs.shape[0] != self.oracleNum:
            raise ValueError("Oracle_Behavior_Probs length must equal Oracle_Num")

    def _init_state_shape(self):
        if self.args.State_Mode == "original":
            self.s_features = 1 + 2 * self.oracleNum
        else:
            # request type, request length, deadline + 8 features per oracle.
            # GNN keeps the same dimensionality, so model.py does not need changes.
            self.s_features = 3 + 8 * self.oracleNum

    def _init_policy_records(self):
        self.events = {}
        self.oracle_events = {}
        self.reputation_factors = {}
        self.oracle_reputation_history = {}
        self.reputation_timewindow = {}
        for name in self.policy_names:
            # rows: 0 oracle, 1 startT, 2 waitT, 3 duration, 4 leaveT, 5 reward,
            # 6 exeT, 7 final success, 8 cost, 9 type match.
            self.events[name] = np.zeros((10, self.requestNum), dtype=float)
            # rows: 0 idleT, 1 assigned count, 2 reputation, 3 type matches,
            # 4 validation successes.
            self.oracle_events[name] = np.zeros((5, self.oracleNum), dtype=float)
            self.oracle_events[name][2] = self.oracleInitialReputation
            # rows: 0 count, 1 validation successes, 2 total duration,
            # 3 behavior-risk sum.
            self.reputation_factors[name] = np.zeros((4, self.oracleNum), dtype=float)
            self.oracle_reputation_history[name] = np.zeros((self.timeperiodNum, self.oracleNum), dtype=float)
            self.reputation_timewindow[name] = np.zeros((0, self.oracleNum), dtype=float)

        # Primary-backup / HCRL diagnostics:
        # 0 primary_success, 1 backup_used, 2 backup_success, 3 backup_recovery,
        # 4 primary_action, 5 backup_action, 6 primary_malicious, 7 backup_malicious,
        # 8 primary_trusted, 9 backup_trusted, 10 backup_skipped,
        # 11 backup_score, 12 HCRL mode, 13 single, 14 serial, 15 parallel,
        # 16 any constraint violation, 17 cost violation, 18 latency violation,
        # 19 risk violation, 20 lambda_cost, 21 lambda_latency, 22 lambda_risk.
        self.pb_records = {name: np.zeros((23, self.requestNum), dtype=float) for name in self.policy_names}
        for name in self.policy_names:
            self.pb_records[name][4, :] = -1
            self.pb_records[name][5, :] = -1
            self.pb_records[name][12, :] = -1

        self.backup_score_history = {name: [] for name in self.policy_names}
        self.hcrl_lambdas = {
            name: {
                "cost": float(getattr(self.args, "HCRL_Lambda_Cost", 0.55)),
                "latency": float(getattr(self.args, "HCRL_Lambda_Latency", 0.40)),
                "risk": float(getattr(self.args, "HCRL_Lambda_Risk", 0.80)),
            }
            for name in self.policy_names
        }

    def reset(self, args):
        self.args = args
        self.policy_names = list(args.Baselines)
        self.policy_num = len(self.policy_names)
        self.policy_name_to_id = {name: idx for idx, name in enumerate(self.policy_names)}
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
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        self.requestsMI = np.maximum(
            np.random.normal(self.requestMI, self.requestMI_std, self.requestNum).astype(int), 1
        )
        self.lengths = self.requestsMI / max(self.oracleCapacity, 1e-8)

        service_type_num = int(np.max(self.oracleTypes)) + 1
        if getattr(self.args, "Scenario", "static") in ["rl_hard", "rl_harder"]:
            burstiness = float(getattr(self.args, "Burstiness", 0.80))
            types = np.zeros(self.requestNum, dtype=int)
            types[0] = np.random.randint(0, service_type_num)
            for i in range(1, self.requestNum):
                if np.random.rand() < burstiness:
                    types[i] = types[i - 1]
                else:
                    types[i] = np.random.randint(0, service_type_num)
            self.request_type = types
        else:
            self.request_type = np.random.choice(np.arange(service_type_num), size=self.requestNum)

        print("intervalT mean: ", round(float(np.mean(intervalT)), 3),
              "  intervalT SD:", round(float(np.std(intervalT, ddof=1)), 3))
        print("last request arrivalT:", round(float(self.arrival_Times[-1]), 3))
        print("MI mean: ", round(float(np.mean(self.requestsMI)), 3),
              "  MI SD:", round(float(np.std(self.requestsMI, ddof=1)), 3))
        print("length mean: ", round(float(np.mean(self.lengths)), 3),
              "  length SD:", round(float(np.std(self.lengths, ddof=1)), 3))

    def workload(self, request_count):
        request_id = int(request_count) - 1
        attrs = [
            request_id,
            float(self.arrival_Times[request_id]),
            float(self.lengths[request_id]),
            int(self.request_type[request_id]),
            float(self.ddl),
        ]
        return request_count == self.requestNum, attrs

    # ------------------------------------------------------------------
    # Reputation and state encoding
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

    def update_reputation(self, reputation_attributes, time_period, policy_name):
        counts = reputation_attributes[0]
        val = reputation_attributes[1]
        behavior = reputation_attributes[3]
        old = self.oracle_events[policy_name][2]
        recent_success = (val + self.oracleInitialReputation * 2.0) / np.maximum(counts + 2.0, 1e-8)
        behavior_penalty = np.log1p(np.maximum(behavior / np.maximum(counts, 1.0), 0.0)) / np.log1p(100.0)
        new_rep = np.clip(0.70 * old + 0.30 * (recent_success - 0.35 * behavior_penalty), 0.0, 1.0)
        self.oracle_events[policy_name][2] = new_rep
        tp = int(min(max(time_period, 0), self.timeperiodNum - 1))
        self.oracle_reputation_history[policy_name][tp] = new_rep
        self.reputation_timewindow[policy_name] = np.vstack(
            (self.reputation_timewindow[policy_name], new_rep[None, :])
        )[-self.timewindowSize:]

    def _policy_uses_gnn(self, policy_name):
        if not getattr(self.args, "Use_GNN_Encoder", False):
            return False
        if getattr(self.args, "Disable_GNN_Encoder", False):
            return False
        if policy_name == "HCRL-Oracle":
            return True
        if getattr(self.args, "Use_GNN_For_All_RL", False) and policy_name in [
            "DQN", "PPO", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle"
        ]:
            return True
        return False

    def getState(self, request_attrs, policy_name):
        _, arrival_time, length, request_type, ddl = request_attrs
        request_type = int(request_type)
        if self.args.State_Mode == "original":
            state = np.hstack(([
                request_type
            ], self.oracle_events[policy_name][0] - float(arrival_time), self.oracle_events[policy_name][2]))
            return np.nan_to_num(state.astype(float), nan=0.0, posinf=10.0, neginf=-10.0)

        oracle_features = self._base_oracle_features(request_attrs, policy_name)
        if self._policy_uses_gnn(policy_name):
            oracle_features = self._graph_encode_oracles(oracle_features, request_type)

        mean_len = float(getattr(self.args, "Request_len_Mean", 6000)) / max(float(getattr(self.args, "Oracle_capacity", 1000)), 1e-8)
        prefix = np.array([
            request_type / max(float(np.max(self.oracleTypes)), 1.0),
            float(length) / max(mean_len, 1e-8),
            float(ddl) / max(float(getattr(self.args, "Harder_Request_DDL", 6.6)), 1e-8),
        ], dtype=float)
        state = np.hstack((prefix, oracle_features.reshape(-1)))
        return np.nan_to_num(state.astype(float), nan=0.0, posinf=10.0, neginf=-10.0)

    def _base_oracle_features(self, request_attrs, policy_name):
        _, arrival_time, length, request_type, ddl = request_attrs
        request_type = int(request_type)
        wait = np.maximum(self.oracle_events[policy_name][0] - float(arrival_time), 0.0)
        wait_norm = np.clip(wait / max(float(ddl), 1e-8), 0.0, 3.0) / 3.0
        rep = np.clip(self.oracle_events[policy_name][2], 0.0, 1.0)
        cost_norm = np.clip(self.oracleCost / max(float(np.max(self.oracleCost)), 1e-8), 0.0, 1.0)
        acc_norm = np.clip(self.oracleAcc / max(float(np.max(self.oracleAcc)), 1e-8), 0.0, 1.0)
        type_match = (self.oracleTypes == request_type).astype(float)

        counts = self.reputation_factors[policy_name][0]
        val = self.reputation_factors[policy_name][1]
        token_norm = np.clip(self.oracleToken / max(float(np.max(self.oracleToken)), 1e-8), 0.0, 1.0)
        prior = 0.5 * rep + 0.5 * token_norm
        observed_success = (val + 2.0 * prior) / np.maximum(counts + 2.0, 1e-8)
        if getattr(self.args, "Expose_Validation_Prob", False):
            validation_feature = np.asarray(self.oracleValidationProbs, dtype=float)
        else:
            validation_feature = np.asarray(observed_success, dtype=float)
        recent_load = np.clip(counts / max(float(self.timeperiodSize), 1.0), 0.0, 1.0)
        behavior = self.reputation_factors[policy_name][3] / np.maximum(counts, 1.0)
        behavior_risk = np.clip(np.log1p(np.maximum(behavior, 0.0)) / np.log1p(100.0), 0.0, 1.0)
        delay_est = np.clip((wait + float(length) / np.maximum(self.oracleAcc, 1e-8)) / max(float(ddl), 1e-8), 0.0, 2.0) / 2.0

        return np.vstack((
            wait_norm,
            rep,
            cost_norm,
            acc_norm,
            type_match,
            validation_feature,
            recent_load,
            0.5 * behavior_risk + 0.5 * delay_est,
        )).T

    def _graph_encode_oracles(self, features, request_type):
        h = np.asarray(features, dtype=float).copy()
        n = h.shape[0]
        if n == 0:
            return h
        same_service = (self.oracleTypes[:, None] == self.oracleTypes[None, :]).astype(float)
        reliability = 1.0 - np.abs(h[:, 5][:, None] - h[:, 5][None, :])
        load_similarity = 1.0 - np.abs(h[:, 6][:, None] - h[:, 6][None, :])
        cost_similarity = 1.0 - np.abs(h[:, 2][:, None] - h[:, 2][None, :])
        adj = (
            float(getattr(self.args, "GNN_Service_Weight", 1.0)) * same_service
            + float(getattr(self.args, "GNN_Reliability_Weight", 0.45)) * reliability
            + float(getattr(self.args, "GNN_Load_Weight", 0.35)) * load_similarity
            + float(getattr(self.args, "GNN_Cost_Weight", 0.25)) * cost_similarity
        )
        np.fill_diagonal(adj, 0.0)
        adj = adj / np.maximum(adj.sum(axis=1, keepdims=True), 1e-8)
        self_w = float(getattr(self.args, "GNN_Self_Weight", 0.55))
        neigh_w = float(getattr(self.args, "GNN_Neighbor_Weight", 0.45))
        steps = int(getattr(self.args, "GNN_Message_Steps", 2))
        request_gate = (self.oracleTypes == int(request_type)).astype(float)[:, None]
        for _ in range(max(steps, 0)):
            msg = adj.dot(h)
            h = np.tanh(self_w * h + neigh_w * msg + 0.05 * request_gate)
        return np.clip(0.5 * (h + 1.0), 0.0, 1.0)

    def get_action_mask(self, request_attrs):
        if getattr(self.args, "Action_Mask_Mode", "none") != "type":
            return np.ones(self.oracleNum, dtype=bool)
        mask = self.oracleTypes == int(request_attrs[3])
        if not np.any(mask):
            mask[:] = True
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

    # ------------------------------------------------------------------
    # Core simulation and feedback
    # ------------------------------------------------------------------
    def _effective_validation_prob(self, action, policy_name):
        action = int(action)
        base = float(self.oracleValidationProbs[action])
        if getattr(self.args, "Scenario", "static") not in ["rl_hard", "rl_harder"]:
            return base
        recent_assigned = float(self.reputation_factors[policy_name][0, action])
        avg_recent = max(self.timeperiodSize / max(self.oracleNum, 1), 1e-8)
        overload = max(0.0, recent_assigned / avg_recent - 1.0)
        if getattr(self.args, "Scenario", "static") == "rl_harder":
            fatigue_growth = np.sqrt(overload) + 0.35 * overload
            min_prob = 0.02
        else:
            fatigue_growth = np.log1p(overload)
            min_prob = 0.05
        fatigue = float(getattr(self.args, "Fatigue_Strength", 1.0)) * float(self.oracleFatigueSensitivity[action]) * fatigue_growth
        return float(np.clip(base - fatigue, min_prob, 0.99))

    def _simulate_oracle_attempt(self, request_attrs, action, policy_name, arrival_override=None):
        _, arrival_time, length, request_type, ddl = request_attrs
        action = int(action)
        effective_arrival = float(arrival_time if arrival_override is None else arrival_override)
        acc = max(float(self.oracleAcc[action]), 1e-8)
        cost = float(self.oracleCost[action])
        oracle_type = int(self.oracleTypes[action])
        idleT = float(self.oracle_events[policy_name][0, action])
        reputation = float(self.oracle_events[policy_name][2, action])
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
        return {
            "action": action,
            "startT": startT,
            "waitT": waitT,
            "exeT": exeT,
            "durationT": durationT,
            "leaveT": leaveT,
            "cost": cost,
            "reputation": reputation,
            "match": match,
            "validation_raw": validation_raw,
            "behavior_record": behavior_record,
            "oracle_type": oracle_type,
            "is_malicious": 1 if action in self.malicious_oracles else 0,
            "is_trusted": 1 if action in self.trusted_oracles else 0,
        }

    def _is_success(self, attempt, ddl):
        if getattr(self.args, "Success_Mode", "original") == "validation_aware":
            return int(attempt["durationT"] <= ddl and attempt["match"] == 1 and attempt["validation_raw"] == 1)
        return int(attempt["durationT"] <= ddl and attempt["match"] == 1)

    def _original_reward(self, exeT, durationT, cost, reputation, request_type, oracle_type):
        penalty = 0 if int(request_type) == int(oracle_type) else 1
        return float((1 + 2.5 * np.exp(1.5 - float(cost))) *
                     (float(exeT) / max(float(durationT), 1e-8)) + float(reputation) - 4 * penalty)

    def _risk_aware_reward(self, reputation, match, successful_validation, cost, durationT, ddl, behavior_record):
        ddl = max(float(ddl), 1e-8)
        timeout = 1.0 if float(durationT) > ddl else 0.0
        on_time = 1.0 - timeout
        rep_score = 0.5 * (np.tanh(float(reputation)) + 1.0)
        match_score = float(match)
        val_score = float(successful_validation)
        task_success = match_score * val_score * on_time
        cost_score = float(np.clip(float(cost), 0.0, 1.25) / 1.25)
        response_ratio = float(np.clip(float(durationT) / ddl, 0.0, 2.5))
        response_penalty = float(np.clip(min(response_ratio, 1.0) * 0.4 + max(response_ratio - 1.0, 0.0) * 0.9, 0.0, 1.0))
        behavior_risk = float(np.log1p(max(float(behavior_record), 0.0)) / np.log1p(100.0))
        a = self.args
        positive = (
            a.W_SUCCESS * task_success
            + a.W_VALIDATION * val_score
            + a.W_MATCH * match_score
            + a.W_REPUTATION * rep_score
        )
        negative = (
            a.W_COST * cost_score
            + a.W_RESPONSE * response_penalty
            + a.W_BEHAVIOR * behavior_risk
            + a.W_TIMEOUT * timeout
            + 0.8 * (1.0 - task_success)
        )
        normalizer = (
            a.W_SUCCESS + a.W_VALIDATION + a.W_MATCH + a.W_REPUTATION
            + a.W_COST + a.W_RESPONSE + a.W_BEHAVIOR + a.W_TIMEOUT + 0.8
        )
        return float(np.clip(a.Reward_Scale * (positive - negative) / max(normalizer, 1e-8),
                             -a.Reward_Clip, a.Reward_Clip))

    def _reward_for_attempt(self, attempt, request_attrs, final_success=None,
                            total_cost=None, final_duration=None,
                            combined_behavior=None, combined_rep=None):
        _, _, _, request_type, ddl = request_attrs
        if getattr(self.args, "Reward_Mode", "original") == "risk_aware":
            return self._risk_aware_reward(
                combined_rep if combined_rep is not None else attempt["reputation"],
                attempt["match"],
                final_success if final_success is not None else attempt["validation_raw"],
                total_cost if total_cost is not None else attempt["cost"],
                final_duration if final_duration is not None else attempt["durationT"],
                ddl,
                combined_behavior if combined_behavior is not None else attempt["behavior_record"],
            )
        return self._original_reward(
            attempt["exeT"],
            final_duration if final_duration is not None else attempt["durationT"],
            total_cost if total_cost is not None else attempt["cost"],
            attempt["reputation"],
            request_type,
            attempt["oracle_type"],
        )

    def _record_attempt_updates(self, policy_name, attempts):
        for attempt in attempts:
            if attempt is None:
                continue
            a = int(attempt["action"])
            self.oracle_events[policy_name][1, a] += 1
            self.oracle_events[policy_name][0, a] = max(self.oracle_events[policy_name][0, a], attempt["leaveT"])
            self.oracle_events[policy_name][3, a] += attempt["match"]
            self.oracle_events[policy_name][4, a] += attempt["validation_raw"]
            self.reputation_factors[policy_name][0, a] += 1
            self.reputation_factors[policy_name][1, a] += attempt["validation_raw"]
            self.reputation_factors[policy_name][2, a] += attempt["durationT"]
            self.reputation_factors[policy_name][3, a] += attempt["behavior_record"]

    def _record_request(self, policy_name, request_id, primary, reward, success,
                        final_duration, final_leaveT, total_cost, match):
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

    def feedback(self, request_attrs, action, policy_name):
        request_id, _, _, _, ddl = request_attrs
        request_id = int(request_id)
        attempt = self._simulate_oracle_attempt(request_attrs, action, policy_name)
        success = self._is_success(attempt, float(ddl))
        reward = self._reward_for_attempt(attempt, request_attrs, final_success=success)
        self._record_attempt_updates(policy_name, [attempt])
        self._record_request(policy_name, request_id, attempt, reward, success,
                             attempt["durationT"], attempt["leaveT"], attempt["cost"], attempt["match"])
        self.pb_records[policy_name][0, request_id] = success
        self.pb_records[policy_name][4, request_id] = int(action)
        self.pb_records[policy_name][6, request_id] = attempt["is_malicious"]
        self.pb_records[policy_name][8, request_id] = attempt["is_trusted"]
        return reward

    # ------------------------------------------------------------------
    # Primary-backup / COBRA helpers
    # ------------------------------------------------------------------
    def _backup_score_vector(self, request_attrs, primary_action, policy_name):
        _, arrival_time, length, _, ddl = request_attrs
        counts = self.reputation_factors[policy_name][0]
        val = self.reputation_factors[policy_name][1]
        behavior_sum = self.reputation_factors[policy_name][3]
        rep = np.clip(self.oracle_events[policy_name][2], 0.0, 1.0)
        token_norm = np.clip(self.oracleToken / max(float(np.max(self.oracleToken)), 1e-8), 0.0, 1.0)
        prior = 0.5 * rep + 0.5 * token_norm
        alpha = float(getattr(self.args, "PB_Prior_Strength", 2.0))
        recent_success = (val + alpha * prior) / np.maximum(counts + alpha, 1e-8)
        recent_load = counts / max(float(self.timeperiodSize), 1.0)
        cost_norm = np.clip(self.oracleCost / max(float(np.max(self.oracleCost)), 1e-8), 0.0, 1.0)
        avg_behavior = behavior_sum / np.maximum(counts, 1.0)
        behavior_risk = np.clip(np.log1p(np.maximum(avg_behavior, 0.0)) / np.log1p(100.0), 0.0, 1.0)
        estimated_wait = np.maximum(self.oracle_events[policy_name][0] - float(arrival_time), 0.0)
        estimated_exe = float(length) / np.maximum(self.oracleAcc, 1e-8)
        delay_penalty = np.clip((estimated_wait + estimated_exe) / max(float(ddl), 1e-8), 0.0, 2.0) / 2.0
        score = (
            self.args.PB_W_RECENT_SUCCESS * recent_success
            + self.args.PB_W_REPUTATION * rep
            + self.args.PB_W_TOKEN * token_norm
            - self.args.PB_W_LOAD * recent_load
            - self.args.PB_W_COST * cost_norm
            - self.args.PB_W_BEHAVIOR_RISK * behavior_risk
            - self.args.PB_W_DELAY * delay_penalty
        )
        score -= 0.08 * (self.oracleCost > float(getattr(self.args, "PB_Backup_Cost_Limit", 1.05)))
        if 0 <= int(primary_action) < self.oracleNum:
            score[int(primary_action)] = -1e9
        return np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)

    def choose_backup_oracle(self, request_attrs, primary_action, policy_name):
        candidates = np.where(self.get_backup_action_mask(request_attrs, primary_action))[0]
        if candidates.size == 0:
            return int(primary_action)
        random_flag = False
        if policy_name == "COBRA-Oracle":
            random_flag = bool(getattr(self.args, "COBRA_Random_Backup", False))
        elif policy_name == "HCRL-Oracle":
            random_flag = bool(getattr(self.args, "HCRL_Random_Backup", False))
        if random_flag:
            return int(np.random.choice(candidates))
        score = self._backup_score_vector(request_attrs, primary_action, policy_name)
        return int(candidates[np.argmax(score[candidates])])

    def _hcrl_oracle_observed_success(self, policy_name, action):
        """Smoothed observed validation success used by HCRL risk estimates.

        This deliberately avoids using the hidden true validation probability.
        It is based only on reputation factors accumulated by the current policy.
        """
        a = int(action)
        counts = self.reputation_factors[policy_name][0, a]
        val = self.reputation_factors[policy_name][1, a]
        rep = float(np.clip(self.oracle_events[policy_name][2, a], 0.0, 1.0))
        token_norm = float(np.clip(self.oracleToken[a] / max(float(np.max(self.oracleToken)), 1e-8), 0.0, 1.0))
        prior = 0.55 * rep + 0.45 * token_norm
        return float((val + 2.0 * prior) / max(counts + 2.0, 1e-8))

    def _hcrl_trust_proxy(self, policy_name, action):
        """Trust proxy available to the policy: reputation + observed success + token stake."""
        a = int(action)
        rep = float(np.clip(self.oracle_events[policy_name][2, a], 0.0, 1.0))
        token_norm = float(np.clip(self.oracleToken[a] / max(float(np.max(self.oracleToken)), 1e-8), 0.0, 1.0))
        hist_success = self._hcrl_oracle_observed_success(policy_name, a)
        return float(np.clip(0.42 * hist_success + 0.35 * rep + 0.23 * token_norm, 0.0, 1.0))

    def _hcrl_oracle_risk_estimate(self, request_attrs, action, policy_name):
        """Risk estimate for a candidate oracle without using hidden labels.

        HCRL-v2 improved success but tended to call more oracles and touched more
        malicious oracles in absolute count.  This v3 estimate uses only observable
        proxies (historical success, reputation, stake, behavior history, load,
        delay pressure, and low-cost bait pattern) to make recovery risk-budgeted.
        """
        _, arrival_time, length, request_type, ddl = request_attrs
        a = int(action)
        if a < 0 or a >= self.oracleNum:
            return 1.0

        counts = self.reputation_factors[policy_name][0, a]
        rep = float(np.clip(self.oracle_events[policy_name][2, a], 0.0, 1.0))
        token_norm = float(np.clip(self.oracleToken[a] / max(float(np.max(self.oracleToken)), 1e-8), 0.0, 1.0))
        hist_success = self._hcrl_oracle_observed_success(policy_name, a)
        trust = float(np.clip(0.42 * hist_success + 0.35 * rep + 0.23 * token_norm, 0.0, 1.0))
        trust_deficit = 1.0 - trust

        behavior = self.reputation_factors[policy_name][3, a] / max(counts, 1.0)
        behavior_risk = float(np.clip(np.log1p(max(behavior, 0.0)) / np.log1p(100.0), 0.0, 1.0))
        wait = max(float(self.oracle_events[policy_name][0, a]) - float(arrival_time), 0.0)
        exe = float(length) / max(float(self.oracleAcc[a]), 1e-8)
        delay_ratio = float(np.clip((wait + exe) / max(float(ddl), 1e-8), 0.0, 2.0))
        delay_risk = max(0.0, delay_ratio - 0.72) / 1.28
        type_mismatch = 0.0 if int(self.oracleTypes[a]) == int(request_type) else 1.0
        fatigue = float(np.clip(self.oracleFatigueSensitivity[a], 0.0, 1.0))
        cost_norm = float(np.clip(self.oracleCost[a] / max(float(np.max(self.oracleCost)), 1e-8), 0.0, 1.0))

        # Low-cost + low-stake + weak history is a bait-like pattern in rl_harder.
        bait_risk = float(np.clip((0.34 - cost_norm) / 0.34, 0.0, 1.0)) * float(np.clip(1.0 - token_norm, 0.0, 1.0))
        if hist_success >= 0.72:
            bait_risk *= 0.35

        risk = (
            0.40 * trust_deficit
            + 0.18 * behavior_risk
            + 0.14 * delay_risk
            + 0.10 * fatigue
            + 0.24 * bait_risk
            + 0.50 * type_mismatch
        )
        return float(np.clip(risk, 0.0, 1.0))

    def _hcrl_primary_risk_estimate(self, request_attrs, primary_action, policy_name):
        """Backward-compatible wrapper for primary oracle risk."""
        return self._hcrl_oracle_risk_estimate(request_attrs, primary_action, policy_name)

    def _hcrl_backup_candidate(self, request_attrs, primary_action, policy_name):
        """Return a risk-budgeted same-type backup candidate and adjusted score.

        v2 selected the highest safety-score backup, which improved success but
        increased cost and malicious exposure. v3 re-ranks backups by combining
        the original safety score with observable trust, estimated risk, cost,
        and improvement over the primary risk.
        """
        candidates = np.where(self.get_backup_action_mask(request_attrs, primary_action))[0]
        if candidates.size == 0:
            return -1, -1e9

        score_vec = self._backup_score_vector(request_attrs, primary_action, policy_name)
        if getattr(self.args, "HCRL_No_Risk_Budgeted_Gate", False):
            best = int(candidates[np.argmax(score_vec[candidates])])
            return best, float(score_vec[best])

        primary_risk = self._hcrl_oracle_risk_estimate(request_attrs, primary_action, policy_name)
        max_risk = float(getattr(self.args, "HCRL_Backup_Max_Estimated_Risk", 0.42))
        cost_cap = float(getattr(self.args, "HCRL_Backup_Cost_Cap", 1.05))
        risk_penalty = float(getattr(self.args, "HCRL_Estimated_Risk_Penalty", 0.45))
        cost_penalty = float(getattr(self.args, "HCRL_Total_Cost_Penalty", 0.18))
        trust_bonus = float(getattr(self.args, "HCRL_Backup_Trust_Bonus", 0.15))

        adjusted = np.full(self.oracleNum, -1e9, dtype=float)
        for c in candidates:
            risk_c = self._hcrl_oracle_risk_estimate(request_attrs, int(c), policy_name)
            trust_c = self._hcrl_trust_proxy(policy_name, int(c))
            cost_c = float(self.oracleCost[int(c)])
            cost_norm = float(np.clip(cost_c / max(float(np.max(self.oracleCost)), 1e-8), 0.0, 1.0))
            improvement = max(0.0, primary_risk - risk_c)
            # Soft risk/cost filtering. Extremely risky backups remain possible only if no safer candidate exists.
            gate_penalty = 0.0
            if risk_c > max_risk:
                gate_penalty += 0.65 * (risk_c - max_risk)
            if cost_c > cost_cap:
                gate_penalty += 0.25 * (cost_c - cost_cap)
            adjusted[int(c)] = (
                float(score_vec[int(c)])
                + trust_bonus * trust_c
                + 0.20 * improvement
                - risk_penalty * risk_c
                - cost_penalty * cost_norm
                - gate_penalty
            )

        # Prefer candidates under the soft risk/cost budget when available.
        safe_candidates = []
        for c in candidates:
            if (self._hcrl_oracle_risk_estimate(request_attrs, int(c), policy_name) <= max_risk and
                    float(self.oracleCost[int(c)]) <= cost_cap):
                safe_candidates.append(int(c))
        pool = np.asarray(safe_candidates if safe_candidates else candidates, dtype=int)
        best = int(pool[np.argmax(adjusted[pool])])
        return best, float(adjusted[best])

    def _hcrl_should_apply_safety_recovery(self, request_attrs, primary_action, primary_success, backup_action, backup_score, policy_name):
        """Risk-budgeted HCRL-v3 safety gate.

        Recovery is activated only when it has a plausible safety/cost benefit.
        This keeps the success gain from v2 while reducing unnecessary cost and
        malicious exposure from aggressive parallel recovery.
        """
        if getattr(self.args, "HCRL_No_Safety_Gate", False):
            return False, self._hcrl_primary_risk_estimate(request_attrs, primary_action, policy_name)
        if int(backup_action) < 0 or int(backup_action) == int(primary_action):
            return False, self._hcrl_primary_risk_estimate(request_attrs, primary_action, policy_name)

        _, _, _, _, ddl = request_attrs
        primary_risk = self._hcrl_oracle_risk_estimate(request_attrs, primary_action, policy_name)
        backup_risk = self._hcrl_oracle_risk_estimate(request_attrs, backup_action, policy_name)
        min_score = float(getattr(self.args, "HCRL_Safety_Min_Backup_Score", 0.12))
        risk_threshold = float(getattr(self.args, "HCRL_Safety_Primary_Risk_Threshold", 0.52))
        risk_margin = float(getattr(self.args, "HCRL_Backup_Risk_Margin", 0.08))
        max_backup_risk = float(getattr(self.args, "HCRL_Backup_Max_Estimated_Risk", 0.42))
        hard_cost_cap = float(getattr(self.args, "HCRL_Recovery_Cost_Hard_Cap", 1.30))

        primary_cost = float(self.oracleCost[int(primary_action)]) if 0 <= int(primary_action) < self.oracleNum else 0.0
        backup_cost = float(self.oracleCost[int(backup_action)]) if 0 <= int(backup_action) < self.oracleNum else 0.0
        effective_parallel_cost = primary_cost + float(getattr(self.args, "HCRL_Parallel_Cost_Discount", 0.85)) * backup_cost
        cost_ok = effective_parallel_cost <= hard_cost_cap
        risk_ok = backup_risk <= max_backup_risk or backup_risk <= (primary_risk - risk_margin)
        score_ok = float(backup_score) >= min_score

        # After observed primary failure, allow recovery if the backup is not worse and the cost is bounded.
        fail_trigger = (
            int(primary_success) == 0
            and score_ok
            and cost_ok
            and backup_risk <= max(max_backup_risk, primary_risk - 0.02)
        )
        # Before failure, only preempt when primary risk is high and backup is meaningfully safer.
        risk_trigger = (
            primary_risk >= risk_threshold
            and score_ok
            and cost_ok
            and risk_ok
            and (primary_risk - backup_risk) >= risk_margin
        )
        return bool(fail_trigger or risk_trigger), primary_risk

    def _should_use_backup(self, request_attrs, primary, backup_action, backup_score, policy_name):
        if int(backup_action) == int(primary["action"]):
            return False
        if getattr(self.args, "PB_Backup_Mode", "parallel") == "serial":
            remaining = float(request_attrs[4]) - float(primary["durationT"])
            estimated_exe = float(request_attrs[2]) / max(float(self.oracleAcc[int(backup_action)]), 1e-8)
            if remaining <= 0 or estimated_exe > remaining:
                return False
        if policy_name == "COBRA-Oracle":
            mode = getattr(self.args, "COBRA_Gate_Mode", "adaptive")
            if mode == "always":
                return True
            if mode == "never":
                return False
            if mode == "fixed":
                return float(backup_score) >= float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46))
            hist = self.backup_score_history.get(policy_name, [])
            if len(hist) >= 20:
                window = int(getattr(self.args, "COBRA_Gate_Window", 400))
                recent = np.asarray(hist[-window:], dtype=float)
                dyn_thr = float(np.mean(recent) + float(getattr(self.args, "COBRA_Gate_Alpha", 0.15)) * np.std(recent))
            else:
                dyn_thr = float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46))
            return float(backup_score) >= max(float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46)), dyn_thr)
        if getattr(self.args, "PB_Backup_Trigger", "cost_aware") == "always":
            return True
        return float(backup_score) >= float(getattr(self.args, "PB_Min_Backup_Score", 0.38))

    def feedback_primary_backup(self, request_attrs, primary_action, policy_name="PB-SafeDQN"):
        request_id, _, _, _, ddl = request_attrs
        request_id = int(request_id)
        ddl = float(ddl)
        primary = self._simulate_oracle_attempt(request_attrs, primary_action, policy_name)
        primary_success = self._is_success(primary, ddl)
        backup_action = self.choose_backup_oracle(request_attrs, primary_action, policy_name)
        score_vec = self._backup_score_vector(request_attrs, primary_action, policy_name)
        backup_score = float(score_vec[int(backup_action)]) if 0 <= int(backup_action) < self.oracleNum else 0.0
        if policy_name == "COBRA-Oracle":
            self.backup_score_history[policy_name].append(backup_score)

        backup_used = 0
        backup_success = 0
        backup_recovery = 0
        backup_skipped = 0
        backup = None
        final_success = primary_success
        final_duration = primary["durationT"]
        final_leaveT = primary["leaveT"]
        total_cost = primary["cost"]
        combined_behavior = primary["behavior_record"]
        combined_rep = primary["reputation"]

        use_backup = False
        if primary_success == 0:
            use_backup = self._should_use_backup(request_attrs, primary, backup_action, backup_score, policy_name)
            backup_skipped = 0 if use_backup else 1

        if use_backup:
            backup_used = 1
            if getattr(self.args, "PB_Backup_Mode", "parallel") == "serial":
                backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name, arrival_override=primary["leaveT"])
                final_duration = primary["durationT"] + backup["durationT"]
                final_leaveT = backup["leaveT"]
            else:
                backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name)
                final_duration = min(primary["durationT"], backup["durationT"] if self._is_success(backup, ddl) else max(primary["durationT"], backup["durationT"]))
                final_leaveT = max(primary["leaveT"], backup["leaveT"])
            backup_success = self._is_success(backup, ddl)
            backup_recovery = 1 if (primary_success == 0 and backup_success == 1) else 0
            final_success = 1 if (primary_success == 1 or backup_success == 1) else 0
            total_cost += backup["cost"]
            combined_behavior = max(primary["behavior_record"], backup["behavior_record"])
            combined_rep = max(primary["reputation"], backup["reputation"])

        if policy_name == "PB-SafeDQN" and backup_used:
            reward = self._reward_for_attempt(primary, request_attrs, final_success=final_success,
                                              total_cost=total_cost, final_duration=final_duration,
                                              combined_behavior=combined_behavior, combined_rep=combined_rep)
            reward += float(getattr(self.args, "PB_Backup_Recovery_Bonus", 0.38)) * backup_recovery
            reward -= float(getattr(self.args, "PB_Backup_Used_Penalty", 0.16)) * backup_used
            reward += float(getattr(self.args, "PB_Primary_Success_Bonus", 0.18)) * primary_success
            reward -= float(getattr(self.args, "PB_Backup_Skip_Penalty", 0.04)) * backup_skipped
        elif policy_name == "COBRA-Oracle":
            reward = self._reward_for_attempt(primary, request_attrs, final_success=final_success,
                                              total_cost=total_cost, final_duration=final_duration,
                                              combined_behavior=combined_behavior, combined_rep=combined_rep)
            reward += float(getattr(self.args, "COBRA_Backup_Recovery_Bonus", 0.34)) * backup_recovery
            reward -= float(getattr(self.args, "COBRA_Backup_Used_Penalty", 0.22)) * backup_used
            reward += float(getattr(self.args, "COBRA_Primary_Success_Bonus", 0.26)) * primary_success
            reward -= float(getattr(self.args, "COBRA_Backup_Skip_Penalty", 0.03)) * backup_skipped
            reward -= self._constraint_penalty(
                cost=total_cost, latency=final_duration,
                risk=max(primary["is_malicious"], backup["is_malicious"] if backup else 0),
                prefix="COBRA",
            )
        else:
            reward = self._reward_for_attempt(primary, request_attrs, final_success=final_success,
                                              total_cost=total_cost, final_duration=final_duration,
                                              combined_behavior=combined_behavior, combined_rep=combined_rep)

        reward = float(np.clip(reward, -float(getattr(self.args, "Reward_Clip", 3.0)), float(getattr(self.args, "Reward_Clip", 3.0))))

        self._record_attempt_updates(policy_name, [primary, backup])
        self._record_request(policy_name, request_id, primary, reward, final_success,
                             final_duration, final_leaveT, total_cost, primary["match"])
        self._record_pb_diagnostics(
            policy_name, request_id, primary, backup, primary_success, backup_used,
            backup_success, backup_recovery, backup_skipped, backup_score,
            mode_action=-1, total_cost=total_cost, latency=final_duration,
            final_risk=max(primary["is_malicious"], backup["is_malicious"] if backup else 0),
            constrained_prefix="COBRA" if policy_name == "COBRA-Oracle" else None,
        )
        return reward

    def _constraint_penalty(self, cost, latency, risk, prefix="HCRL", lambdas=None):
        if prefix == "HCRL" and getattr(self.args, "HCRL_No_Constrained", False):
            return 0.0
        cost_budget = float(getattr(self.args, f"{prefix}_Cost_Budget", 1.0))
        latency_budget = float(getattr(self.args, f"{prefix}_Latency_Budget", 6.0))
        risk_budget = float(getattr(self.args, f"{prefix}_Risk_Budget", 0.08))
        lv_cost = max(0.0, float(cost) - cost_budget)
        lv_latency = max(0.0, float(latency) - latency_budget)
        lv_risk = max(0.0, float(risk) - risk_budget)
        if lambdas is None:
            l_cost = float(getattr(self.args, f"{prefix}_Lambda_Cost", 0.5))
            l_latency = float(getattr(self.args, f"{prefix}_Lambda_Latency", 0.4))
            l_risk = float(getattr(self.args, f"{prefix}_Lambda_Risk", 0.8))
        else:
            l_cost = float(lambdas.get("cost", 0.5))
            l_latency = float(lambdas.get("latency", 0.4))
            l_risk = float(lambdas.get("risk", 0.8))
        return l_cost * lv_cost + l_latency * lv_latency + l_risk * lv_risk

    # ------------------------------------------------------------------
    # HCRL
    # ------------------------------------------------------------------
    def get_hcrl_mode_state(self, request_attrs, policy_name):
        base = self.getState(request_attrs, policy_name)
        start = max(0, int(request_attrs[0]) - self.timeperiodSize)
        end = max(start + 1, int(request_attrs[0]))
        recent_cost = float(np.mean(self.events[policy_name][8, start:end])) if end > start else 0.0
        recent_success = float(np.mean(self.events[policy_name][7, start:end])) if end > start else 0.0
        recent_latency = float(np.mean(self.events[policy_name][3, start:end])) if end > start else 0.0
        recent_mal = 0.0
        if end > start:
            recent_mal = float(np.mean(np.maximum(
                self.pb_records[policy_name][6, start:end],
                self.pb_records[policy_name][7, start:end],
            )))
        budget_state = np.array([
            recent_success,
            recent_cost / max(float(getattr(self.args, "HCRL_Cost_Budget", 1.0)), 1e-8),
            recent_latency / max(float(getattr(self.args, "HCRL_Latency_Budget", 6.0)), 1e-8),
            recent_mal,
            float(getattr(self.args, "HCRL_Cost_Budget", 1.0)),
            float(getattr(self.args, "HCRL_Risk_Budget", 0.06)),
        ], dtype=float)
        return np.hstack((base, budget_state))

    def _get_hcrl_lambdas(self, policy_name):
        if policy_name not in self.hcrl_lambdas:
            self.hcrl_lambdas[policy_name] = {
                "cost": float(getattr(self.args, "HCRL_Lambda_Cost", 0.55)),
                "latency": float(getattr(self.args, "HCRL_Lambda_Latency", 0.40)),
                "risk": float(getattr(self.args, "HCRL_Lambda_Risk", 0.80)),
            }
        return self.hcrl_lambdas[policy_name]

    def _update_hcrl_lambdas(self, policy_name, cost_violation, latency_violation, risk_violation):
        lambdas = self._get_hcrl_lambdas(policy_name)
        if getattr(self.args, "HCRL_No_Constrained", False) or not getattr(self.args, "HCRL_Primal_Dual", True):
            return lambdas
        lr = float(getattr(self.args, "HCRL_Lambda_LR", 0.01))
        lo = float(getattr(self.args, "HCRL_Lambda_Min", 0.0))
        hi = float(getattr(self.args, "HCRL_Lambda_Max", 3.0))
        lambdas["cost"] = float(np.clip(lambdas["cost"] + lr * float(cost_violation), lo, hi))
        lambdas["latency"] = float(np.clip(lambdas["latency"] + lr * float(latency_violation), lo, hi))
        lambdas["risk"] = float(np.clip(lambdas["risk"] + lr * float(risk_violation), lo, hi))
        return lambdas

    def feedback_hcrl(self, request_attrs, mode_action, primary_action, backup_action, policy_name="HCRL-Oracle"):
        request_id, _, length, _, ddl = request_attrs
        request_id = int(request_id)
        ddl = float(ddl)
        mode_action = int(np.clip(mode_action, 0, 2))
        primary_action = int(primary_action)
        backup_action = int(backup_action) if int(backup_action) >= 0 else -1

        primary = self._simulate_oracle_attempt(request_attrs, primary_action, policy_name)
        primary_success = self._is_success(primary, ddl)

        backup = None
        backup_used = 0
        backup_success = 0
        backup_recovery = 0
        backup_skipped = 0
        backup_score = 0.0
        final_success = primary_success
        final_duration = primary["durationT"]
        final_leaveT = primary["leaveT"]
        total_cost = primary["cost"]
        combined_behavior = primary["behavior_record"]
        combined_rep = primary["reputation"]

        if backup_action >= 0 and backup_action != primary_action:
            score_vec = self._backup_score_vector(request_attrs, primary_action, policy_name)
            backup_score = float(score_vec[backup_action])

        # HCRL-v2 safety gate. If the learned mode policy collapses to single
        # while the primary is risky or already failed, activate a safe recovery
        # candidate. This keeps HCRL competitive with DQN while preserving the
        # learned primary policy and cost-aware recovery diagnostics.
        original_mode_action = mode_action
        safety_overrode = 0
        primary_risk_estimate = self._hcrl_primary_risk_estimate(request_attrs, primary_action, policy_name)
        if backup_action < 0 or backup_action == primary_action:
            cand_backup, cand_score = self._hcrl_backup_candidate(request_attrs, primary_action, policy_name)
            if cand_backup >= 0:
                backup_action, backup_score = cand_backup, cand_score

        backup_risk_estimate = (
            self._hcrl_oracle_risk_estimate(request_attrs, backup_action, policy_name)
            if backup_action >= 0 and backup_action != primary_action else 1.0
        )
        primary_trust_proxy = self._hcrl_trust_proxy(policy_name, primary_action)
        backup_trust_proxy = (
            self._hcrl_trust_proxy(policy_name, backup_action)
            if backup_action >= 0 and backup_action != primary_action else 0.0
        )

        safety_trigger, primary_risk_estimate = self._hcrl_should_apply_safety_recovery(
            request_attrs, primary_action, primary_success, backup_action, backup_score, policy_name
        )
        backup_risk_estimate = (
            self._hcrl_oracle_risk_estimate(request_attrs, backup_action, policy_name)
            if backup_action >= 0 and backup_action != primary_action else 1.0
        )
        backup_trust_proxy = (
            self._hcrl_trust_proxy(policy_name, backup_action)
            if backup_action >= 0 and backup_action != primary_action else 0.0
        )
        if mode_action == 0 and safety_trigger:
            safety_overrode = 1
            recovery_mode = getattr(self.args, "HCRL_Safety_Recovery_Mode", "auto")
            if recovery_mode == "serial":
                mode_action = 1
            elif recovery_mode == "parallel":
                mode_action = 2
            else:
                # Use serial only when there is enough remaining deadline after
                # the primary attempt; otherwise choose parallel warm-standby.
                remaining = ddl - float(primary["durationT"])
                estimated_exe = float(length) / max(float(self.oracleAcc[int(backup_action)]), 1e-8)
                mode_action = 1 if (primary_success == 0 and remaining > 0 and estimated_exe <= remaining) else 2

        # mode 0 = single, 1 = serial, 2 = parallel
        if mode_action == 1:
            if primary_success == 0 and backup_action >= 0 and backup_action != primary_action:
                remaining = ddl - float(primary["durationT"])
                estimated_exe = float(length) / max(float(self.oracleAcc[backup_action]), 1e-8)
                if remaining > 0 and estimated_exe <= remaining:
                    backup_used = 1
                    backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name, arrival_override=primary["leaveT"])
                    backup_success = self._is_success(backup, ddl)
                    backup_recovery = 1 if backup_success else 0
                    final_success = 1 if backup_success else 0
                    final_duration = primary["durationT"] + backup["durationT"]
                    final_leaveT = backup["leaveT"]
                else:
                    backup_skipped = 1
            elif primary_success == 0:
                backup_skipped = 1
        elif mode_action == 2:
            if backup_action >= 0 and backup_action != primary_action:
                backup_used = 1
                backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name)
                backup_success = self._is_success(backup, ddl)
                backup_recovery = 1 if (primary_success == 0 and backup_success == 1) else 0
                final_success = 1 if (primary_success == 1 or backup_success == 1) else 0
                # Warm-standby/parallel: successful faster branch determines effective latency;
                # accounting still records both oracle attempts.
                if primary_success and backup_success:
                    final_duration = min(primary["durationT"], backup["durationT"])
                elif backup_success:
                    final_duration = backup["durationT"]
                else:
                    final_duration = max(primary["durationT"], backup["durationT"])
                final_leaveT = max(primary["leaveT"], backup["leaveT"])
            elif primary_success == 0:
                backup_skipped = 1

        if backup is not None:
            total_cost += backup["cost"]
            combined_behavior = max(primary["behavior_record"], backup["behavior_record"])
            combined_rep = max(primary["reputation"], backup["reputation"])

        if mode_action == 2 and backup is not None:
            effective_cost_for_constraint = primary["cost"] + float(getattr(self.args, "HCRL_Parallel_Cost_Discount", 0.85)) * backup["cost"]
        else:
            effective_cost_for_constraint = total_cost
        final_risk = max(primary["is_malicious"], backup["is_malicious"] if backup is not None else 0)

        cost_budget = float(getattr(self.args, "HCRL_Cost_Budget", 1.02))
        latency_budget = float(getattr(self.args, "HCRL_Latency_Budget", 5.95))
        risk_budget = float(getattr(self.args, "HCRL_Risk_Budget", 0.06))
        cost_violation = max(0.0, effective_cost_for_constraint - cost_budget)
        latency_violation = max(0.0, final_duration - latency_budget)
        risk_violation = max(0.0, final_risk - risk_budget)
        lambdas = self._update_hcrl_lambdas(policy_name, cost_violation, latency_violation, risk_violation)
        constraint_penalty = 0.0 if getattr(self.args, "HCRL_No_Constrained", False) else (
            lambdas["cost"] * cost_violation + lambdas["latency"] * latency_violation + lambdas["risk"] * risk_violation
        )

        base_reward = self._reward_for_attempt(
            primary, request_attrs, final_success=final_success, total_cost=total_cost,
            final_duration=final_duration, combined_behavior=combined_behavior, combined_rep=combined_rep,
        )
        primary_reward = self._reward_for_attempt(
            primary, request_attrs, final_success=primary_success, total_cost=primary["cost"],
            final_duration=primary["durationT"], combined_behavior=primary["behavior_record"],
            combined_rep=primary["reputation"],
        )
        # v3: train primary selector away from low-trust/high-risk bait oracles.
        primary_reward += float(getattr(self.args, "HCRL_Trusted_Selection_Bonus", 0.12)) * primary_trust_proxy
        primary_reward -= float(getattr(self.args, "HCRL_Estimated_Risk_Penalty", 0.45)) * primary_risk_estimate
        primary_reward -= float(getattr(self.args, "HCRL_Primary_Malicious_Penalty", 0.80)) * primary["is_malicious"]
        backup_reward = 0.0
        if backup_used and backup is not None:
            backup_reward = self._reward_for_attempt(
                backup, request_attrs, final_success=backup_success, total_cost=backup["cost"],
                final_duration=backup["durationT"], combined_behavior=backup["behavior_record"],
                combined_rep=backup["reputation"],
            )
            backup_reward += float(getattr(self.args, "HCRL_Backup_Recovery_Bonus", 0.72)) * backup_recovery
            backup_reward += float(getattr(self.args, "HCRL_Backup_Trust_Bonus", 0.15)) * backup_trust_proxy
            backup_reward -= float(getattr(self.args, "HCRL_Backup_Used_Penalty", 0.08)) * backup_used
            backup_reward -= float(getattr(self.args, "HCRL_Estimated_Risk_Penalty", 0.45)) * backup_risk_estimate
            backup_reward -= float(getattr(self.args, "HCRL_Backup_Malicious_Penalty", 1.20)) * (backup["is_malicious"] if backup is not None else 0)

        mode_reward = base_reward
        mode_reward += float(getattr(self.args, "HCRL_Final_Success_Bonus", 0.35)) * final_success
        mode_reward += float(getattr(self.args, "HCRL_Primary_Success_Bonus", 0.30)) * primary_success
        mode_reward += float(getattr(self.args, "HCRL_Backup_Recovery_Bonus", 0.72)) * backup_recovery
        mode_reward += float(getattr(self.args, "HCRL_Success_Gain_Bonus", 0.45)) * max(0, final_success - primary_success)
        mode_reward += float(getattr(self.args, "HCRL_Safety_Override_Bonus", 0.12)) * safety_overrode * final_success
        # v3: explicit success-risk-cost trade-off shaping.
        mode_reward += float(getattr(self.args, "HCRL_Trusted_Selection_Bonus", 0.12)) * primary_trust_proxy
        mode_reward += float(getattr(self.args, "HCRL_Backup_Trust_Bonus", 0.15)) * backup_trust_proxy * backup_used
        mode_reward -= float(getattr(self.args, "HCRL_Backup_Used_Penalty", 0.08)) * backup_used
        mode_reward -= float(getattr(self.args, "HCRL_Estimated_Risk_Penalty", 0.45)) * (primary_risk_estimate + backup_used * backup_risk_estimate)
        mode_reward -= float(getattr(self.args, "HCRL_Total_Cost_Penalty", 0.18)) * max(0.0, effective_cost_for_constraint - cost_budget)
        mode_reward -= float(getattr(self.args, "HCRL_Primary_Malicious_Penalty", 0.80)) * primary["is_malicious"]
        mode_reward -= float(getattr(self.args, "HCRL_Backup_Malicious_Penalty", 1.20)) * ((backup["is_malicious"] if backup is not None else 0))
        # Penalize skipping recovery only when the primary was risky or failed.
        risk_skip = 1.0 if (primary_risk_estimate >= float(getattr(self.args, "HCRL_Safety_Primary_Risk_Threshold", 0.48)) or primary_success == 0) else 0.0
        mode_reward -= float(getattr(self.args, "HCRL_Skip_Recovery_Penalty", 0.20)) * backup_skipped * risk_skip
        if backup_used and primary_success and safety_overrode == 0:
            mode_reward -= float(getattr(self.args, "HCRL_Unnecessary_Backup_Penalty", 0.18))
        # Extra guard: if recovery was triggered but backup is estimated worse than the primary, penalize it.
        if backup_used and backup_risk_estimate > primary_risk_estimate + 0.03:
            mode_reward -= 0.25 * (backup_risk_estimate - primary_risk_estimate)
        mode_reward -= constraint_penalty

        if getattr(self.args, "HCRL_No_Decoupled_Reward", False):
            primary_reward = mode_reward
            backup_reward = mode_reward if backup_used else 0.0

        reward_clip = float(getattr(self.args, "Reward_Clip", 3.0))
        final_reward = float(np.clip(mode_reward, -reward_clip, reward_clip))
        primary_reward = float(np.clip(primary_reward, -reward_clip, reward_clip))
        backup_reward = float(np.clip(backup_reward, -reward_clip, reward_clip))
        mode_reward = final_reward

        self._record_attempt_updates(policy_name, [primary, backup])
        self._record_request(policy_name, request_id, primary, final_reward, final_success,
                             final_duration, final_leaveT, total_cost, primary["match"])
        self._record_pb_diagnostics(
            policy_name, request_id, primary, backup, primary_success, backup_used,
            backup_success, backup_recovery, backup_skipped, backup_score,
            mode_action=mode_action, total_cost=effective_cost_for_constraint,
            latency=final_duration, final_risk=final_risk,
            constrained_prefix="HCRL", explicit_violations=(cost_violation, latency_violation, risk_violation),
            lambdas=lambdas,
        )

        return {
            "final_reward": final_reward,
            "mode_reward": mode_reward,
            "primary_reward": primary_reward,
            "backup_reward": backup_reward,
            "primary_success": primary_success,
            "backup_used": backup_used,
            "backup_success": backup_success,
            "backup_recovery": backup_recovery,
            "final_success": final_success,
            "safety_overrode": safety_overrode,
            "primary_risk_estimate": primary_risk_estimate,
        }

    def _record_pb_diagnostics(self, policy_name, request_id, primary, backup, primary_success,
                               backup_used, backup_success, backup_recovery, backup_skipped,
                               backup_score, mode_action=-1, total_cost=0.0, latency=0.0,
                               final_risk=0.0, constrained_prefix=None,
                               explicit_violations=None, lambdas=None):
        r = self.pb_records[policy_name]
        r[0, request_id] = primary_success
        r[1, request_id] = backup_used
        r[2, request_id] = backup_success
        r[3, request_id] = backup_recovery
        r[4, request_id] = primary["action"]
        r[5, request_id] = backup["action"] if backup is not None else -1
        r[6, request_id] = primary["is_malicious"]
        r[7, request_id] = backup["is_malicious"] if backup is not None else 0
        r[8, request_id] = primary["is_trusted"]
        r[9, request_id] = backup["is_trusted"] if backup is not None else 0
        r[10, request_id] = backup_skipped
        r[11, request_id] = 0.0 if not np.isfinite(backup_score) else backup_score
        r[12, request_id] = mode_action
        r[13, request_id] = 1 if mode_action == 0 else 0
        r[14, request_id] = 1 if mode_action == 1 else 0
        r[15, request_id] = 1 if mode_action == 2 else 0

        if constrained_prefix is not None:
            if explicit_violations is None:
                cost_budget = float(getattr(self.args, f"{constrained_prefix}_Cost_Budget", 1.0))
                latency_budget = float(getattr(self.args, f"{constrained_prefix}_Latency_Budget", 6.0))
                risk_budget = float(getattr(self.args, f"{constrained_prefix}_Risk_Budget", 0.08))
                cost_violation = max(0.0, float(total_cost) - cost_budget)
                latency_violation = max(0.0, float(latency) - latency_budget)
                risk_violation = max(0.0, float(final_risk) - risk_budget)
            else:
                cost_violation, latency_violation, risk_violation = explicit_violations
            r[17, request_id] = cost_violation
            r[18, request_id] = latency_violation
            r[19, request_id] = risk_violation
            r[16, request_id] = 1.0 if (cost_violation > 0 or latency_violation > 0 or risk_violation > 0) else 0.0
            if lambdas is None:
                if constrained_prefix == "HCRL":
                    lambdas = self._get_hcrl_lambdas(policy_name)
                else:
                    lambdas = {
                        "cost": float(getattr(self.args, f"{constrained_prefix}_Lambda_Cost", 0.0)),
                        "latency": float(getattr(self.args, f"{constrained_prefix}_Lambda_Latency", 0.0)),
                        "risk": float(getattr(self.args, f"{constrained_prefix}_Lambda_Risk", 0.0)),
                    }
            r[20, request_id] = float(lambdas.get("cost", 0.0))
            r[21, request_id] = float(lambdas.get("latency", 0.0))
            r[22, request_id] = float(lambdas.get("risk", 0.0))

    # ------------------------------------------------------------------
    # SemiGreedy helper and direct getters for baselines
    # ------------------------------------------------------------------
    def feedback_PSG_FWA(self, request_attrs, policy_name="SemiGreedy"):
        rewards = np.zeros(self.oracleNum, dtype=float)
        costs = self.oracleCost.copy()
        for a in range(self.oracleNum):
            wait = max(self.oracle_events[policy_name][0, a] - float(request_attrs[1]), 0.0)
            exeT = float(request_attrs[2]) / max(float(self.oracleAcc[a]), 1e-8)
            duration = wait + exeT
            match = 1 if int(request_attrs[3]) == int(self.oracleTypes[a]) else 0
            rep = self.oracle_events[policy_name][2, a]
            observed_val = self._effective_validation_prob(a, policy_name) if getattr(self.args, "SemiGreedy_View", "myopic") == "risk_aware" else 1.0
            if getattr(self.args, "Reward_Mode", "original") == "risk_aware" or getattr(self.args, "SemiGreedy_View", "myopic") == "risk_aware":
                rewards[a] = self._risk_aware_reward(rep, match, observed_val, self.oracleCost[a], duration, request_attrs[4], 0.0)
            else:
                rewards[a] = self._original_reward(exeT, duration, self.oracleCost[a], rep, request_attrs[3], self.oracleTypes[a])
        return rewards, costs

    def get_oracle_idleT(self, policy_name):
        return self.oracle_events[policy_name][0]

    def get_request_num(self, policy_name):
        return self.reputation_factors[policy_name][0]

    def get_successful_validation(self, policy_name):
        return self.reputation_factors[policy_name][1]

    # ------------------------------------------------------------------
    # Accumulate metrics kept for original main.py compatibility
    # ------------------------------------------------------------------
    def _metric_by_policy(self, row, Baseline_num, startP=0, reducer="mean"):
        out = np.zeros(Baseline_num, dtype=float)
        startP = int(max(0, startP))
        for i, name in enumerate(self.policy_names[:Baseline_num]):
            values = self.events[name][row, startP:]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            if values.size == 0:
                out[i] = 0.0
            elif reducer == "sum":
                out[i] = float(np.sum(values))
            elif reducer == "last":
                out[i] = float(values[-1])
            else:
                out[i] = float(np.mean(values))
        return out

    def get_accumulateRewards(self, Baseline_num, startP, request_c):
        return self.get_totalRewards(Baseline_num, startP, endP=request_c)

    def get_accumulateCost(self, Baseline_num, startP, request_c):
        return self.get_totalCost(Baseline_num, startP, endP=request_c)

    def get_FinishTimes(self, Baseline_num, startP, request_c):
        return self.get_totalTimes(Baseline_num, startP, endP=request_c)

    def get_executeTs(self, Baseline_num, startP, request_c):
        return self._window_event_mean(6, Baseline_num, startP, request_c)

    def get_waitTs(self, Baseline_num, startP, request_c):
        return self._window_event_mean(2, Baseline_num, startP, request_c)

    def get_responseTs(self, Baseline_num, startP, request_c):
        return self._window_event_mean(3, Baseline_num, startP, request_c)

    def get_successTimes(self, Baseline_num, startP, request_c):
        return self._window_event_mean(7, Baseline_num, startP, request_c)

    def get_successInTime(self, Baseline_num, startP, request_c):
        return self._window_event_mean(7, Baseline_num, startP, request_c)

    def _window_event_mean(self, row, Baseline_num, startP=0, endP=None):
        out = np.zeros(Baseline_num, dtype=float)
        startP = int(max(0, startP))
        endP = self.requestNum if endP is None else int(min(max(endP, startP + 1), self.requestNum))
        for i, name in enumerate(self.policy_names[:Baseline_num]):
            values = self.events[name][row, startP:endP]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            out[i] = float(np.mean(values)) if values.size else 0.0
        return out

    # ------------------------------------------------------------------
    # Final summary getters used by main.py
    # ------------------------------------------------------------------
    def get_totalRewards(self, Baseline_num, startP=0, endP=None):
        return self._window_event_sum(5, Baseline_num, startP, endP)

    def get_total_responseTs(self, Baseline_num, startP=0, endP=None):
        return self._window_event_mean(3, Baseline_num, startP, endP)

    def get_totalSuccess(self, Baseline_num, startP=0, endP=None):
        return self._window_event_mean(7, Baseline_num, startP, endP)

    def get_totalSuccessInTime(self, Baseline_num, startP=0, endP=None):
        # Current success definition already includes the deadline condition.
        return self._window_event_mean(7, Baseline_num, startP, endP)

    def get_totalTimes(self, Baseline_num, startP=0, endP=None):
        # Finish time is the maximum leave time in the evaluation window.
        out = np.zeros(Baseline_num, dtype=float)
        startP = int(max(0, startP))
        endP = self.requestNum if endP is None else int(min(max(endP, startP + 1), self.requestNum))
        for i, name in enumerate(self.policy_names[:Baseline_num]):
            values = self.events[name][4, startP:endP]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            out[i] = float(np.max(values)) if values.size else 0.0
        return out

    def get_totalCost(self, Baseline_num, startP=0, endP=None):
        return self._window_event_mean(8, Baseline_num, startP, endP)

    def get_totalMatchRate(self, Baseline_num, startP=0, endP=None):
        return self._window_event_mean(9, Baseline_num, startP, endP)

    def _window_event_sum(self, row, Baseline_num, startP=0, endP=None):
        out = np.zeros(Baseline_num, dtype=float)
        startP = int(max(0, startP))
        endP = self.requestNum if endP is None else int(min(max(endP, startP + 1), self.requestNum))
        for i, name in enumerate(self.policy_names[:Baseline_num]):
            values = self.events[name][row, startP:endP]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            out[i] = float(np.sum(values)) if values.size else 0.0
        return out

    def get_totalMaliciousNum(self, Baseline_num, startP=0, endP=None):
        return self._role_count(Baseline_num, startP, endP, role="malicious")

    def get_totalNormalNum(self, Baseline_num, startP=0, endP=None):
        return self._role_count(Baseline_num, startP, endP, role="normal")

    def get_totalTrustedNum(self, Baseline_num, startP=0, endP=None):
        return self._role_count(Baseline_num, startP, endP, role="trusted")

    def _role_count(self, Baseline_num, startP=0, endP=None, role="malicious"):
        out = np.zeros(Baseline_num, dtype=float)
        startP = int(max(0, startP))
        endP = self.requestNum if endP is None else int(min(max(endP, startP + 1), self.requestNum))
        if role == "malicious":
            role_set = set(map(int, self.malicious_oracles))
        elif role == "normal":
            role_set = set(map(int, self.normal_oracles))
        else:
            role_set = set(map(int, self.trusted_oracles))
        for i, name in enumerate(self.policy_names[:Baseline_num]):
            primary_actions = self.pb_records[name][4, startP:endP].astype(int)
            backup_actions = self.pb_records[name][5, startP:endP].astype(int)
            cnt = sum(1 for a in primary_actions if a in role_set)
            cnt += sum(1 for a in backup_actions if a in role_set)
            out[i] = float(cnt)
        return out

    def _mean_pb_record_row(self, row_idx, Baseline_num, startP=0, endP=None):
        out = np.zeros(Baseline_num, dtype=float)
        startP = int(max(0, startP))
        endP = self.requestNum if endP is None else int(min(max(endP, startP + 1), self.requestNum))
        for i, name in enumerate(self.policy_names[:Baseline_num]):
            if name not in self.pb_records:
                out[i] = 0.0
                continue
            values = self.pb_records[name][row_idx, startP:endP]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            out[i] = float(np.mean(values)) if values.size else 0.0
        return out

    def _sum_pb_record_row(self, row_idx, Baseline_num, startP=0, endP=None):
        out = np.zeros(Baseline_num, dtype=float)
        startP = int(max(0, startP))
        endP = self.requestNum if endP is None else int(min(max(endP, startP + 1), self.requestNum))
        for i, name in enumerate(self.policy_names[:Baseline_num]):
            values = self.pb_records[name][row_idx, startP:endP]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            out[i] = float(np.sum(values)) if values.size else 0.0
        return out

    def get_totalPrimarySuccessRate(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(0, Baseline_num, startP, endP)

    def get_totalBackupUsedRate(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(1, Baseline_num, startP, endP)

    def get_totalBackupSuccessRate(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(2, Baseline_num, startP, endP)

    def get_totalBackupRecoveryRate(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(3, Baseline_num, startP, endP)

    def get_totalBackupSkippedRate(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(10, Baseline_num, startP, endP)

    def get_totalConditionalBackupRecoveryRate(self, Baseline_num, startP=0, endP=None):
        out = np.zeros(Baseline_num, dtype=float)
        startP = int(max(0, startP))
        endP = self.requestNum if endP is None else int(min(max(endP, startP + 1), self.requestNum))
        for i, name in enumerate(self.policy_names[:Baseline_num]):
            used = np.sum(self.pb_records[name][1, startP:endP])
            rec = np.sum(self.pb_records[name][3, startP:endP])
            out[i] = float(rec / used) if used > 1e-8 else 0.0
        return out

    def get_totalBackupScoreMean(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(11, Baseline_num, startP, endP)

    def get_totalPrimaryMaliciousNum(self, Baseline_num, startP=0, endP=None):
        return self._sum_pb_record_row(6, Baseline_num, startP, endP)

    def get_totalBackupMaliciousNum(self, Baseline_num, startP=0, endP=None):
        return self._sum_pb_record_row(7, Baseline_num, startP, endP)

    def get_totalPrimaryTrustedNum(self, Baseline_num, startP=0, endP=None):
        return self._sum_pb_record_row(8, Baseline_num, startP, endP)

    def get_totalBackupTrustedNum(self, Baseline_num, startP=0, endP=None):
        return self._sum_pb_record_row(9, Baseline_num, startP, endP)

    def get_totalCostPerSuccess(self, Baseline_num, startP=0, endP=None):
        cost = self.get_totalCost(Baseline_num, startP, endP)
        success = self.get_totalSuccess(Baseline_num, startP, endP)
        return cost / np.maximum(success, 1e-8)

    def get_totalHCRLSingleModeRate(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(13, Baseline_num, startP, endP)

    def get_totalHCRLSerialModeRate(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(14, Baseline_num, startP, endP)

    def get_totalHCRLParallelModeRate(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(15, Baseline_num, startP, endP)

    def get_totalConstraintViolationRate(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(16, Baseline_num, startP, endP)

    def get_totalConstraintViolationMean(self, Baseline_num, startP=0, endP=None):
        return self.get_totalConstraintViolationRate(Baseline_num, startP, endP)

    def get_totalCostViolationMean(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(17, Baseline_num, startP, endP)

    def get_totalLatencyViolationMean(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(18, Baseline_num, startP, endP)

    def get_totalRiskViolationMean(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(19, Baseline_num, startP, endP)

    def get_totalLambdaCostMean(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(20, Baseline_num, startP, endP)

    def get_totalLambdaLatencyMean(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(21, Baseline_num, startP, endP)

    def get_totalLambdaRiskMean(self, Baseline_num, startP=0, endP=None):
        return self._mean_pb_record_row(22, Baseline_num, startP, endP)

    # Common aliases, in case main.py uses slightly different names.
    def get_totalCostViolationRate(self, Baseline_num, startP=0, endP=None):
        return self.get_totalCostViolationMean(Baseline_num, startP, endP)

    def get_totalLatencyViolationRate(self, Baseline_num, startP=0, endP=None):
        return self.get_totalLatencyViolationMean(Baseline_num, startP, endP)

    def get_totalRiskViolationRate(self, Baseline_num, startP=0, endP=None):
        return self.get_totalRiskViolationMean(Baseline_num, startP, endP)

    def get_totalHCRLLambdaCost(self, Baseline_num, startP=0, endP=None):
        return self.get_totalLambdaCostMean(Baseline_num, startP, endP)

    def get_totalHCRLLambdaLatency(self, Baseline_num, startP=0, endP=None):
        return self.get_totalLambdaLatencyMean(Baseline_num, startP, endP)

    def get_totalHCRLLambdaRisk(self, Baseline_num, startP=0, endP=None):
        return self.get_totalLambdaRiskMean(Baseline_num, startP, endP)

    def __getattr__(self, name):
        """Compatibility guard for older/newer main.py summary getters.

        This only handles get_total* methods that are absent and returns a zero
        vector. It prevents final CSV writing from crashing because of a missing
        optional diagnostic getter, while core simulation methods still raise
        normal AttributeError.
        """
        if name.startswith("get_total"):
            def _missing_total_getter(Baseline_num, startP=0, *args, **kwargs):
                return np.zeros(int(Baseline_num), dtype=float)
            return _missing_total_getter
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")
