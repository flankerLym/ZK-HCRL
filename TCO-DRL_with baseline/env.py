import numpy as np
from scipy import stats


class SchedulingEnv:
    """Scalable oracle-selection simulation environment.

    This replacement keeps the original TCO-DRL metrics and baseline names, but removes
    hard-coded 15-oracle assumptions so that Oracle_Num can scale to 30/60/105+.
    """

    def __init__(self, args):
        self.args = args
        self.policy_names = list(args.Baselines)
        self.policy_num = len(self.policy_names)
        self.policy_name_to_id = {name: idx for idx, name in enumerate(self.policy_names)}

        # Oracle settings
        self.oracleTypes = np.array(args.Oracle_Type, dtype=int)
        self.oracleNum = int(args.Oracle_Num)
        assert self.oracleNum == len(self.oracleTypes), "Oracle_Num must equal len(Oracle_Type)"
        self.oracleCapacity = args.Oracle_capacity
        self.actionNum = self.oracleNum
        self.oracleInitialReputation = args.Oracle_Initial_Reputation
        self.oracleAcc = np.array(args.Oracle_Acc, dtype=float)
        self.oracleCost = np.array(args.Oracle_Cost, dtype=float)
        self.oracleToken = np.array(args.Oracle_Tokens, dtype=float)
        self.oracleBehaviorProbs = np.array(args.Oracle_Behavior_Probs, dtype=float)
        self.oracleValidationProbs = np.array(args.Oracle_Validation_Probs, dtype=float)
        self.oracleFatigueSensitivity = np.array(getattr(args, "Oracle_Fatigue_Sensitivity", [0.0] * self.oracleNum), dtype=float)
        self.malicious_oracles = list(args.Malicious_Oracle_Index)
        self.normal_oracles = list(args.Normal_Oracle_Index)
        self.trusted_oracles = list(args.Trusted_Oracle_Index)

        # Request settings
        self.requestMI = args.Request_len_Mean
        self.requestMI_std = args.Request_len_Std
        self.requestNum = int(args.Request_Num)
        self.lamda = args.lamda
        self.ddl = args.Request_ddl
        self.noise_probability = args.Noise_Probability
        self.noise_delay = args.Noise_Delay

        # State dimension
        if args.State_Mode == "original":
            # request type + oracle wait time + oracle reputation
            self.s_features = 1 + 2 * self.oracleNum
        else:
            # request type, request length, ddl + wait + reputation + cost + acc +
            # type_match + base validation prob + recent success rate + recent load
            self.s_features = 3 + 8 * self.oracleNum

        # Reputation settings
        self.timewindowSize = args.Time_Window_Size
        self.timeperiodSize = args.Time_Period_Size
        self.timeperiodNum = int(self.requestNum / self.timeperiodSize) + 2

        self.arrival_Times = np.zeros(self.requestNum)
        self.requestsMI = np.zeros(self.requestNum)
        self.lengths = np.zeros(self.requestNum)
        self.request_type = np.zeros(self.requestNum, dtype=int)

        self._init_policy_records()
        self.gen_workload(self.lamda)

    def _init_policy_records(self):
        self.events = {}
        self.oracle_events = {}
        self.reputation_factors = {}
        self.oracle_reputation_history = {}
        self.reputation_timewindow = {}

        for name in self.policy_names:
            # request events rows:
            # 0 oracle id, 1 start time, 2 wait time, 3 duration, 4 leave time,
            # 5 reward, 6 base exeT, 7 success, 8 cost, 9 success_without_type
            self.events[name] = np.zeros((10, self.requestNum))
            # oracle events rows:
            # 0 idleT, 1 assigned request num, 2 reputation, 3 match num, 4 successful validation num
            self.oracle_events[name] = np.zeros((5, self.oracleNum))
            self.oracle_events[name][2] = self.oracleInitialReputation
            self.reputation_factors[name] = np.zeros((4, self.oracleNum))
            self.oracle_reputation_history[name] = np.zeros((self.timeperiodNum, self.oracleNum))
            self.reputation_timewindow[name] = np.zeros((0, self.oracleNum))

        # Primary-backup diagnostic records. Rows:
        # 0 primary_success, 1 backup_used, 2 backup_success, 3 backup_recovery,
        # 4 primary_action, 5 backup_action, 6 primary_malicious, 7 backup_malicious,
        # 8 primary_trusted, 9 backup_trusted, 10 backup_skipped_by_trigger,
        # 11 selected_backup_score. Values remain zero for non-PB policies.
        self.pb_records = {name: np.zeros((12, self.requestNum)) for name in self.policy_names}
        for name in self.policy_names:
            self.pb_records[name][4, :] = -1
            self.pb_records[name][5, :] = -1
        # Recent backup utility histories for adaptive COBRA gates.
        self.backup_score_history = {name: [] for name in self.policy_names}

    def gen_workload(self, lamda):
        intervalT = stats.expon.rvs(scale=1 / lamda * 60, size=self.requestNum)
        print("intervalT mean: ", round(np.mean(intervalT), 3),
              '  intervalT SD:', round(np.std(intervalT, ddof=1), 3))
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        print('last request arrivalT:', round(self.arrival_Times[-1], 3))

        self.requestsMI = np.random.normal(self.requestMI, self.requestMI_std, self.requestNum).astype(int)
        print("MI mean: ", round(np.mean(self.requestsMI), 3),
              '  MI SD:', round(np.std(self.requestsMI, ddof=1), 3))
        self.lengths = self.requestsMI / self.oracleCapacity
        print("length mean: ", round(np.mean(self.lengths), 3),
              '  length SD:', round(np.std(self.lengths, ddof=1), 3))

        service_type_num = int(np.max(self.oracleTypes)) + 1
        if getattr(self.args, "Scenario", "static") in ["rl_hard", "rl_harder"]:
            # Bursty correlated requests: a one-step greedy policy tends to repeatedly
            # hit the same cheap same-type oracle, triggering fatigue. RL agents can
            # observe recent load/success features and learn to distribute selections.
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
            probs = [1.0 / service_type_num] * service_type_num
            self.request_type = np.random.choice(list(range(service_type_num)), size=self.requestNum, p=probs)

    def reset(self, args):
        self.args = args
        self.policy_names = list(args.Baselines)
        self.policy_num = len(self.policy_names)
        self.policy_name_to_id = {name: idx for idx, name in enumerate(self.policy_names)}

        self.oracleTypes = np.array(args.Oracle_Type, dtype=int)
        self.oracleNum = int(args.Oracle_Num)
        self.actionNum = self.oracleNum
        self.oracleAcc = np.array(args.Oracle_Acc, dtype=float)
        self.oracleCost = np.array(args.Oracle_Cost, dtype=float)
        self.oracleToken = np.array(args.Oracle_Tokens, dtype=float)
        self.oracleBehaviorProbs = np.array(args.Oracle_Behavior_Probs, dtype=float)
        self.oracleValidationProbs = np.array(args.Oracle_Validation_Probs, dtype=float)
        self.oracleFatigueSensitivity = np.array(getattr(args, "Oracle_Fatigue_Sensitivity", [0.0] * self.oracleNum), dtype=float)
        self.malicious_oracles = list(args.Malicious_Oracle_Index)
        self.normal_oracles = list(args.Normal_Oracle_Index)
        self.trusted_oracles = list(args.Trusted_Oracle_Index)

        self.requestNum = int(args.Request_Num)
        self.ddl = args.Request_ddl
        self.noise_probability = args.Noise_Probability
        self.noise_delay = args.Noise_Delay
        self.timewindowSize = args.Time_Window_Size
        self.timeperiodSize = args.Time_Period_Size
        self.timeperiodNum = int(self.requestNum / self.timeperiodSize) + 2

        if args.State_Mode == "original":
            self.s_features = 1 + 2 * self.oracleNum
        else:
            self.s_features = 3 + 8 * self.oracleNum

        self.arrival_Times = np.zeros(self.requestNum)
        self.requestsMI = np.zeros(self.requestNum)
        self.lengths = np.zeros(self.requestNum)
        self.request_type = np.zeros(self.requestNum, dtype=int)
        self._init_policy_records()
        self.gen_workload(args.lamda)

    def reset_reputation_factors(self):
        for name in self.policy_names:
            if name != "BLOR":
                self.reputation_factors[name] = np.zeros((4, self.oracleNum))

    def reset_reputation_factors_BLOR(self):
        if "BLOR" in self.policy_names:
            self.reputation_factors["BLOR"] = np.zeros((4, self.oracleNum))

    def workload(self, request_count):
        request_id = request_count - 1
        request_attributes = [
            request_id,
            self.arrival_Times[request_id],
            self.lengths[request_id],
            int(self.request_type[request_id]),
            self.ddl,
        ]
        return request_count == self.requestNum, request_attributes

    def _original_reward(self, exeT, durationT, cost, reputation, request_type, oracle_type):
        penalty = 0 if request_type == oracle_type else 1
        return (1 + 2.5 * np.exp(1.5 - cost)) * (exeT / max(durationT, 1e-8)) + reputation - 4 * penalty

    def _risk_aware_reward(self, reputation, match, successful_validation, cost, durationT, ddl, behavior_record):
        """Bounded success-aligned risk-aware reward (tuned).

        Design goal:
        1) keep every one-step reward small and clipped;
        2) align the objective with validation-aware success;
        3) prevent RA-DDQN from becoming too conservative by over-selecting
           expensive trusted oracles that cause queueing/timeouts.

        Therefore the dominant positive term is direct task success
        (on-time + type match + validation), while reputation is a small
        regularizer. Cost, latency, timeout and abnormal behavior are explicit
        penalties. This keeps reward values interpretable and avoids the
        previous "high reward but lower success_rate" mismatch.
        """
        args = self.args

        ddl = max(float(ddl), 1e-8)
        timeout = 1.0 if durationT > ddl else 0.0
        on_time = 1.0 - timeout

        rep_score = 0.5 * (np.tanh(float(reputation)) + 1.0)
        match_score = float(match)
        val_score = float(successful_validation)
        task_success = float(match_score * val_score * on_time)

        cost_score = float(np.clip(cost, 0.0, 1.25) / 1.25)
        response_ratio = float(np.clip(durationT / ddl, 0.0, 2.5))
        # Softer below deadline, rapidly worse after deadline.
        response_penalty = min(response_ratio, 1.0) * 0.4 + max(response_ratio - 1.0, 0.0) * 0.9
        response_penalty = float(np.clip(response_penalty, 0.0, 1.0))
        behavior_risk = float(np.log1p(max(float(behavior_record), 0.0)) / np.log1p(100.0))

        positive = (
            args.W_SUCCESS * task_success
            + args.W_VALIDATION * val_score
            + args.W_MATCH * match_score
            + args.W_REPUTATION * rep_score
        )
        negative = (
            args.W_COST * cost_score
            + args.W_RESPONSE * response_penalty
            + args.W_BEHAVIOR * behavior_risk
            + args.W_TIMEOUT * timeout
            + 0.8 * (1.0 - task_success)
        )

        normalizer = (
            args.W_SUCCESS + args.W_VALIDATION + args.W_MATCH + args.W_REPUTATION
            + args.W_COST + args.W_RESPONSE + args.W_BEHAVIOR + args.W_TIMEOUT
            + 0.8
        )
        raw = (positive - negative) / max(normalizer, 1e-8)
        reward = float(args.Reward_Scale * raw)
        return float(np.clip(reward, -args.Reward_Clip, args.Reward_Clip))

    def _effective_validation_prob(self, action, policy_name):
        """Dynamic validation probability under rl_hard.

        In rl_hard, low-cost bait oracles fatigue when over-used within the current
        reputation period. This makes the environment unfavorable to one-step greedy
        policies that always choose the cheapest matching oracle.
        """
        base_prob = float(self.oracleValidationProbs[action])
        if getattr(self.args, "Scenario", "static") not in ["rl_hard", "rl_harder"]:
            return base_prob
        recent_assigned = float(self.reputation_factors[policy_name][0, action])
        avg_recent = max(self.timeperiodSize / max(self.oracleNum, 1), 1e-6)
        load_ratio = recent_assigned / avg_recent
        # no penalty below average load. rl_harder uses a stronger delayed trap:
        # fatigue grows faster after repeated selection during bursty traffic.
        overload = max(0.0, load_ratio - 1.0)
        scenario = getattr(self.args, "Scenario", "static")
        if scenario == "rl_harder":
            fatigue_growth = np.sqrt(overload) + 0.35 * overload
            min_prob = 0.02
        else:
            fatigue_growth = np.log1p(overload)
            min_prob = 0.05
        fatigue = float(getattr(self.args, "Fatigue_Strength", 1.0)) * float(self.oracleFatigueSensitivity[action]) * fatigue_growth
        return float(np.clip(base_prob - fatigue, min_prob, 0.99))

    def get_action_mask(self, request_attrs):
        """Boolean mask for type-constrained RL action space."""
        if getattr(self.args, "Action_Mask_Mode", "none") != "type":
            return np.ones(self.oracleNum, dtype=bool)
        request_type = int(request_attrs[3])
        mask = self.oracleTypes == request_type
        if not np.any(mask):
            mask[:] = True
        return mask.astype(bool)

    def _simulate_oracle_attempt(self, request_attrs, action, policy_name, arrival_override=None):
        """Simulate one oracle attempt without writing request-level metrics.

        This helper is used by PB-SafeDQN, where a request may involve a primary
        and a backup oracle. It returns the attempt-level timing, validation,
        behavior and role information. The caller decides whether the request is
        finally successful and then writes the combined result into self.events.
        """
        request_id, arrival_time, length, request_type, ddl = request_attrs
        request_type = int(request_type)
        action = int(action)
        effective_arrival = float(arrival_time if arrival_override is None else arrival_override)

        acc = self.oracleAcc[action]
        cost = self.oracleCost[action]
        oracle_type = int(self.oracleTypes[action])
        validation_prob = self._effective_validation_prob(action, policy_name)
        behavior_probs = self.oracleBehaviorProbs[action]

        idleT = self.oracle_events[policy_name][0, action]
        reputation = self.oracle_events[policy_name][2, action]
        exeT = length / acc
        waitT = max(idleT - effective_arrival, 0.0)
        startT = effective_arrival + waitT
        exe_time = (length * 1.05) / acc if action in self.malicious_oracles else exeT
        if np.random.rand() < self.noise_probability:
            real_exeT = exe_time + self.noise_delay
        else:
            real_exeT = exe_time
        durationT = waitT + real_exeT
        leaveT = startT + real_exeT
        match = 1 if request_type == oracle_type else 0
        validation_raw = 1 if np.random.rand() < validation_prob else 0
        behavior_record = np.random.choice([0, 1, 5, 100], p=behavior_probs)
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

    def _backup_score_vector(self, request_attrs, primary_action, policy_name):
        """Observable backup utility for PB-SafeDQN.

        This score deliberately avoids privileged true validation probability.
        It combines historical validation success, reputation, stake tokens,
        recent load, observed abnormal behavior, cost and estimated delay.
        Higher score means the backup is more likely to recover a failed primary
        at acceptable cost and latency.
        """
        arrival_time = float(request_attrs[1])
        length = float(request_attrs[2])
        ddl = max(float(request_attrs[4]), 1e-8)

        req_num = self.reputation_factors[policy_name][0]
        val_num = self.reputation_factors[policy_name][1]
        behavior_sum = self.reputation_factors[policy_name][3]

        reputations = self.oracle_events[policy_name][2]
        rep_norm = 0.5 * (np.tanh(reputations) + 1.0)
        token_norm = np.clip(self.oracleToken / max(float(np.max(self.oracleToken)), 1e-8), 0.0, 1.0)

        # Bayesian-style cold-start smoothing: before many observations, use a
        # weak prior from reputation and stake, not true validation probability.
        prior = 0.5 * rep_norm + 0.5 * token_norm
        alpha = float(getattr(self.args, "PB_Prior_Strength", 2.0))
        recent_success = (val_num + alpha * prior) / np.maximum(req_num + alpha, 1e-8)

        recent_load = req_num / max(self.timeperiodSize, 1)
        cost_norm = np.clip(self.oracleCost / max(float(np.max(self.oracleCost)), 1e-8), 0.0, 1.0)

        avg_behavior = behavior_sum / np.maximum(req_num, 1.0)
        behavior_risk = np.log1p(np.maximum(avg_behavior, 0.0)) / np.log1p(100.0)
        behavior_risk = np.clip(behavior_risk, 0.0, 1.0)

        estimated_wait = np.maximum(self.oracle_events[policy_name][0] - arrival_time, 0.0)
        estimated_exe = length / np.maximum(self.oracleAcc, 1e-8)
        delay_penalty = np.clip((estimated_wait + estimated_exe) / ddl, 0.0, 2.0) / 2.0

        score = (
            self.args.PB_W_RECENT_SUCCESS * recent_success
            + self.args.PB_W_REPUTATION * rep_norm
            + self.args.PB_W_TOKEN * token_norm
            - self.args.PB_W_LOAD * recent_load
            - self.args.PB_W_COST * cost_norm
            - self.args.PB_W_BEHAVIOR_RISK * behavior_risk
            - self.args.PB_W_DELAY * delay_penalty
        )
        # Discourage very expensive backup nodes unless their observed quality is clearly better.
        score = score - 0.08 * (self.oracleCost > self.args.PB_Backup_Cost_Limit)
        score = np.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
        return score

    def choose_backup_oracle(self, request_attrs, primary_action, policy_name):
        """Choose the best same-type backup oracle using observable safety score."""
        mask = self.get_action_mask(request_attrs).astype(bool)
        candidates = np.where(mask)[0]
        candidates = np.array([c for c in candidates if int(c) != int(primary_action)], dtype=int)
        if candidates.size == 0:
            candidates = np.array([c for c in range(self.oracleNum) if int(c) != int(primary_action)], dtype=int)
        if candidates.size == 0:
            return int(primary_action)
        if policy_name == "COBRA-Oracle" and getattr(self.args, "COBRA_Random_Backup", False):
            return int(np.random.choice(candidates))
        score = self._backup_score_vector(request_attrs, primary_action, policy_name)
        return int(candidates[np.argmax(score[candidates])])

    def _should_use_backup(self, request_attrs, primary, backup_action, backup_score, policy_name):
        """Decide whether a backup should be invoked.

        PB-SafeDQN keeps the previous fixed/always gate. COBRA-Oracle uses an
        adaptive utility gate: a backup is triggered only if its observable
        utility exceeds a recent dynamic threshold and the constrained
        cost/latency/risk terms are not obviously unfavorable.
        """
        if int(backup_action) == int(primary["action"]):
            return False

        # Serial backup cannot help when primary already consumes almost all deadline.
        if getattr(self.args, "PB_Backup_Mode", "parallel") == "serial":
            ddl = float(request_attrs[4])
            length = float(request_attrs[2])
            estimated_exe = length / max(float(self.oracleAcc[backup_action]), 1e-8)
            remaining = ddl - float(primary["durationT"])
            if remaining <= 0 or estimated_exe > remaining:
                return False

        if policy_name == "COBRA-Oracle":
            mode = getattr(self.args, "COBRA_Gate_Mode", "adaptive")
            if mode == "always":
                return True
            if mode == "never":
                return False

            # Soft budget pre-screen: do not call backup if it is clearly too
            # expensive and not very high utility. This prevents "always backup"
            # behavior under difficult traces.
            backup_cost = float(self.oracleCost[int(backup_action)])
            cost_budget = float(getattr(self.args, "COBRA_Cost_Budget", 1.00))
            if backup_cost > cost_budget * 1.35 and float(backup_score) < getattr(self.args, "COBRA_Min_Backup_Score", 0.46) + 0.08:
                return False

            if mode == "fixed":
                return float(backup_score) >= float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46))

            hist = self.backup_score_history.get(policy_name, [])
            window = int(getattr(self.args, "COBRA_Gate_Window", 400))
            if len(hist) >= 20:
                recent = np.asarray(hist[-window:], dtype=float)
                dyn_thr = float(np.mean(recent) + float(getattr(self.args, "COBRA_Gate_Alpha", 0.15)) * np.std(recent))
            else:
                dyn_thr = float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46))
            threshold = max(float(getattr(self.args, "COBRA_Min_Backup_Score", 0.46)), dyn_thr)
            return float(backup_score) >= threshold

        # Existing PB-SafeDQN behavior.
        if getattr(self.args, "PB_Backup_Trigger", "cost_aware") == "always":
            return True
        return float(backup_score) >= float(getattr(self.args, "PB_Min_Backup_Score", 0.38))

    def feedback_primary_backup(self, request_attrs, primary_action, policy_name="PB-SafeDQN"):
        """Primary-backup failover feedback for PB-SafeDQN.

        A Dueling Double DQN selects the primary oracle. If the primary attempt
        fails validation-aware success, a backup oracle is selected by an
        observable reputation-load-cost safety rule. In parallel mode, the
        backup represents a warm-standby oracle whose response can recover the
        request without serially doubling the latency. In serial mode, the
        backup starts after the primary attempt. The final event records the
        combined request outcome and cost.
        """
        if policy_name not in self.policy_names:
            raise ValueError(f"Unknown policy_name: {policy_name}")
        request_id, arrival_time, length, request_type, ddl = request_attrs
        request_id = int(request_id)
        ddl = float(ddl)

        primary = self._simulate_oracle_attempt(request_attrs, primary_action, policy_name)
        primary_success = 1 if (primary["durationT"] <= ddl and primary["match"] == 1 and primary["validation_raw"] == 1) else 0

        backup_used = 0
        backup_success = 0
        backup_recovery = 0
        backup_skipped = 0
        backup = None
        backup_action = -1
        backup_score = 0.0
        final_duration = primary["durationT"]
        final_leaveT = primary["leaveT"]
        total_cost = primary["cost"]
        final_success = primary_success
        combined_behavior = primary["behavior_record"]
        combined_rep = primary["reputation"]

        if primary_success == 0:
            backup_action = self.choose_backup_oracle(request_attrs, primary["action"], policy_name)
            backup_scores = self._backup_score_vector(request_attrs, primary["action"], policy_name)
            backup_score = float(backup_scores[int(backup_action)])
            if self._should_use_backup(request_attrs, primary, backup_action, backup_score, policy_name):
                backup_used = 1
                if getattr(self.args, "PB_Backup_Mode", "parallel") == "serial":
                    backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name, arrival_override=primary["leaveT"])
                    final_duration = primary["durationT"] + backup["durationT"]
                    final_leaveT = backup["leaveT"]
                else:
                    backup = self._simulate_oracle_attempt(request_attrs, backup_action, policy_name, arrival_override=arrival_time)
                    final_duration = max(primary["durationT"], backup["durationT"])
                    final_leaveT = float(arrival_time) + final_duration
                total_cost += backup["cost"]
                combined_behavior = max(float(primary["behavior_record"]), float(backup["behavior_record"]))
                combined_rep = 0.5 * (float(primary["reputation"]) + float(backup["reputation"]))
                backup_success = 1 if (final_duration <= ddl and backup["match"] == 1 and backup["validation_raw"] == 1) else 0
                backup_recovery = 1 if backup_success == 1 else 0
                final_success = 1 if backup_success == 1 else 0
            else:
                backup_skipped = 1

        # The final request is type-matched if the successful route was matched;
        # with type action masking this should be true, but keep it explicit.
        final_match = primary["match"] if primary_success else (backup["match"] if backup is not None else primary["match"])
        # Final system-level reward stored in events. For COBRA, this is a
        # constrained reliability-cost objective. The reward returned to the
        # primary Q-network can be decoupled below so backup recovery does not
        # hide primary mistakes.
        final_reward = self._risk_aware_reward(
            combined_rep,
            final_match,
            final_success,
            total_cost,
            final_duration,
            ddl,
            combined_behavior,
        )

        if policy_name == "COBRA-Oracle":
            final_reward += self.args.COBRA_Primary_Success_Bonus * primary_success
            final_reward += self.args.COBRA_Backup_Recovery_Bonus * backup_recovery
            final_reward -= self.args.COBRA_Backup_Used_Penalty * backup_used
            final_reward -= self.args.COBRA_Backup_Skip_Penalty * backup_skipped

            # Soft Lagrangian-style constraints: penalize excessive recovery
            # overhead so COBRA cannot win by simply calling more oracles.
            cost_violation = max(0.0, float(total_cost) - float(self.args.COBRA_Cost_Budget))
            latency_violation = max(0.0, float(final_duration) - float(self.args.COBRA_Latency_Budget)) / max(float(ddl), 1e-8)
            risk_indicator = max(float(primary["is_malicious"]), float(backup["is_malicious"]) if backup is not None else 0.0)
            risk_violation = max(0.0, risk_indicator - float(self.args.COBRA_Risk_Budget))
            final_reward -= self.args.COBRA_Lambda_Cost * cost_violation
            final_reward -= self.args.COBRA_Lambda_Latency * latency_violation
            final_reward -= self.args.COBRA_Lambda_Risk * risk_violation

            primary_train_reward = self._risk_aware_reward(
                primary["reputation"],
                primary["match"],
                primary_success,
                primary["cost"],
                primary["durationT"],
                ddl,
                primary["behavior_record"],
            )
            primary_train_reward += self.args.COBRA_Primary_Success_Bonus * primary_success
            primary_train_reward -= self.args.COBRA_Primary_Malicious_Penalty * primary["is_malicious"]
            primary_train_reward -= 0.12 * (1.0 - primary_success)
            train_reward = final_reward if getattr(self.args, "COBRA_No_Decoupled_Reward", False) else primary_train_reward
        else:
            final_reward += self.args.PB_Primary_Success_Bonus * primary_success
            final_reward += self.args.PB_Backup_Recovery_Bonus * backup_recovery
            final_reward -= self.args.PB_Backup_Used_Penalty * backup_used
            final_reward -= self.args.PB_Backup_Skip_Penalty * backup_skipped
            train_reward = final_reward

        reward = float(np.clip(final_reward, -self.args.Reward_Clip, self.args.Reward_Clip))
        train_reward = float(np.clip(train_reward, -self.args.Reward_Clip, self.args.Reward_Clip))

        # Write final combined request event.
        ev = self.events[policy_name]
        ev[0, request_id] = primary["action"]
        ev[1, request_id] = primary["startT"]
        ev[2, request_id] = primary["waitT"]
        ev[3, request_id] = final_duration
        ev[4, request_id] = final_leaveT
        ev[5, request_id] = reward
        ev[6, request_id] = primary["exeT"]
        ev[7, request_id] = final_success
        ev[8, request_id] = total_cost
        ev[9, request_id] = 1 if final_duration <= ddl else 0

        # Update primary oracle records.
        for attempt in [primary, backup] if backup is not None else [primary]:
            action = int(attempt["action"])
            oe = self.oracle_events[policy_name]
            oe[1, action] += 1
            oe[0, action] = max(oe[0, action], attempt["leaveT"])
            oe[3, action] += attempt["match"]
            # Count raw validation successes for reputation, not final request success.
            oe[4, action] += attempt["validation_raw"]

            rf = self.reputation_factors[policy_name]
            rf[0, action] += 1
            rf[1, action] += attempt["validation_raw"]
            rf[2, action] += attempt["durationT"]
            rf[3, action] += attempt["behavior_record"]

        pb = self.pb_records[policy_name]
        pb[0, request_id] = primary_success
        pb[1, request_id] = backup_used
        pb[2, request_id] = backup_success
        pb[3, request_id] = backup_recovery
        pb[4, request_id] = primary["action"]
        pb[5, request_id] = backup_action
        pb[6, request_id] = primary["is_malicious"]
        pb[7, request_id] = backup["is_malicious"] if backup is not None else 0
        pb[8, request_id] = primary["is_trusted"]
        pb[9, request_id] = backup["is_trusted"] if backup is not None else 0
        pb[10, request_id] = backup_skipped
        pb[11, request_id] = backup_score
        if primary_success == 0:
            self.backup_score_history.setdefault(policy_name, []).append(float(backup_score))

        return train_reward

    def feedback(self, request_attrs, action, policy_name):
        if policy_name not in self.policy_names:
            raise ValueError(f"Unknown policy_name: {policy_name}")
        if action < 0 or action >= self.oracleNum:
            raise IndexError(f"action {action} out of bounds for {self.oracleNum} oracles")

        request_id, arrival_time, length, request_type, ddl = request_attrs
        request_id = int(request_id)
        request_type = int(request_type)
        action = int(action)

        acc = self.oracleAcc[action]
        cost = self.oracleCost[action]
        oracle_type = int(self.oracleTypes[action])
        validation_prob = self._effective_validation_prob(action, policy_name)
        behavior_probs = self.oracleBehaviorProbs[action]

        idleT = self.oracle_events[policy_name][0, action]
        reputation = self.oracle_events[policy_name][2, action]

        exeT = length / acc
        if idleT <= arrival_time:
            waitT = 0.0
            startT = arrival_time
        else:
            waitT = idleT - arrival_time
            startT = idleT

        if action in self.malicious_oracles:
            exe_time = (length * 1.05) / acc
        else:
            exe_time = exeT

        if np.random.rand() < self.noise_probability:
            real_exeT = exe_time + self.noise_delay
        else:
            real_exeT = exe_time

        durationT = waitT + real_exeT
        leaveT = startT + real_exeT
        new_idleT = leaveT

        match = 1 if request_type == oracle_type else 0
        success_without_type = 1 if durationT <= ddl else 0
        successful_validation = 1 if np.random.rand() < validation_prob and durationT <= ddl else 0
        if self.args.Success_Mode == "validation_aware":
            success = 1 if durationT <= ddl and match == 1 and successful_validation == 1 else 0
        else:
            success = 1 if durationT <= ddl and match == 1 else 0
        behavior_record = np.random.choice([0, 1, 5, 100], p=behavior_probs)

        original_reward = self._original_reward(exeT, durationT, cost, reputation, request_type, oracle_type)
        if self.args.Reward_Mode == "risk_aware" or policy_name == "RA-DDQN":
            reward = self._risk_aware_reward(reputation, match, successful_validation, cost, durationT, ddl, behavior_record)
        else:
            reward = original_reward

        ev = self.events[policy_name]
        ev[0, request_id] = action
        ev[1, request_id] = startT
        ev[2, request_id] = waitT
        ev[3, request_id] = durationT
        ev[4, request_id] = leaveT
        ev[5, request_id] = reward
        ev[6, request_id] = exeT
        ev[7, request_id] = success
        ev[8, request_id] = cost
        ev[9, request_id] = success_without_type

        oe = self.oracle_events[policy_name]
        oe[1, action] += 1
        oe[0, action] = new_idleT
        oe[3, action] += match
        oe[4, action] += successful_validation

        rf = self.reputation_factors[policy_name]
        rf[0, action] += 1
        rf[1, action] += successful_validation
        rf[2, action] += durationT
        rf[3, action] += behavior_record

        return reward

    def feedback_PSG_FWA(self, request_attrs, policy_name="SemiGreedy"):
        arrival_time = request_attrs[1]
        length = request_attrs[2]
        request_type = int(request_attrs[3])
        rewards = np.zeros(self.oracleNum)
        idleT = self.oracle_events[policy_name][0]
        reputation = self.oracle_events[policy_name][2]

        for action in range(self.oracleNum):
            waitT = max(idleT[action] - arrival_time, 0)
            exeT = length / self.oracleAcc[action]
            if action in self.malicious_oracles:
                exe_time = (length * 1.05) / self.oracleAcc[action]
            else:
                exe_time = exeT
            durationT = waitT + exe_time
            oracle_type = int(self.oracleTypes[action])
            if self.args.SemiGreedy_View == "risk_aware":
                # Expected one-step risk-aware score. This gives SemiGreedy access to
                # validation statistics; the default myopic view intentionally does not.
                match = 1 if request_type == oracle_type else 0
                expected_validation = self._effective_validation_prob(action, policy_name) if durationT <= request_attrs[4] else 0.0
                expected_behavior = float(np.dot(self.oracleBehaviorProbs[action], np.array([0, 1, 5, 100], dtype=float)))
                rewards[action] = self._risk_aware_reward(
                    reputation[action], match, expected_validation, self.oracleCost[action], durationT, request_attrs[4], expected_behavior
                )
            else:
                rewards[action] = self._original_reward(
                    exeT, durationT, self.oracleCost[action], reputation[action], request_type, oracle_type
                )
        return rewards, self.oracleCost

    def get_reputation_factors(self, policy_name):
        total_requests = self.reputation_factors[policy_name][0]
        successful_validation_requests = self.reputation_factors[policy_name][1]
        total_response_time = self.reputation_factors[policy_name][2]
        behavior_records = self.reputation_factors[policy_name][3]

        success_rate = np.zeros(self.oracleNum)
        average_response_time = np.zeros(self.oracleNum)
        response_time_score = np.zeros(self.oracleNum)

        average_requests = self.timeperiodSize / max(self.oracleNum, 1)
        relative_response_frequency = total_requests / average_requests if average_requests > 0 else 0
        for i in range(self.oracleNum):
            success_rate[i] = successful_validation_requests[i] / total_requests[i] if total_requests[i] > 0 else 0
            average_response_time[i] = total_response_time[i] / total_requests[i] if total_requests[i] > 0 else 0
            response_time_score[i] = self.ddl / average_response_time[i] if average_response_time[i] > 0 else 0
        reliability_score = (relative_response_frequency * 0.2) + (success_rate * 0.4) + (response_time_score * 0.4)
        behavior_score = behavior_records
        avg_tokens = max(float(np.mean(self.oracleToken)), 1e-8)
        staked_tokens_score = self.oracleToken / avg_tokens
        return [reliability_score, behavior_score, staked_tokens_score]

    def initial_reputation(self):
        for name in self.policy_names:
            self.oracle_reputation_history[name][0] = self.oracleInitialReputation
            self.reputation_timewindow[name] = np.append(
                self.reputation_timewindow[name], [self.oracle_reputation_history[name][0]], axis=0
            )
            self.oracle_events[name][2] = self.oracleInitialReputation

    def update_reputation(self, reputation_attributes, current_period, policy_name):
        reliability_score, behavior_score, staked_tokens_score = reputation_attributes
        base_reputation = (reliability_score * 0.4) - (behavior_score * 0.4) + (staked_tokens_score * 0.2)
        reputation = np.zeros(self.oracleNum)

        if current_period <= 1 or current_period > self.timeperiodNum:
            return

        tw = self.reputation_timewindow[policy_name]
        for idx in range(tw.shape[0]):
            k = idx + 2
            time_factor = np.tanh(0.6 / k)
            reputation += time_factor * tw[idx]
        reputation += base_reputation * np.tanh(0.6 / 1)

        self.oracle_reputation_history[policy_name][current_period - 1] = reputation
        self.oracle_events[policy_name][2] = reputation

        new_reputation = np.array(reputation).reshape(1, -1)
        if self.reputation_timewindow[policy_name].shape[0] < self.timewindowSize:
            self.reputation_timewindow[policy_name] = np.vstack((new_reputation, self.reputation_timewindow[policy_name]))
        else:
            self.reputation_timewindow[policy_name] = np.vstack((new_reputation, self.reputation_timewindow[policy_name][:-1, :]))

    def get_oracle_idleT(self, policy_name):
        return self.oracle_events[policy_name][0, :]

    def get_oracle_reputation(self, policy_name):
        return self.oracle_events[policy_name][2, :]

    def get_successful_validation(self, policy_name):
        return np.array(self.reputation_factors[policy_name][1, :])

    def get_request_num(self, policy_name):
        return np.array(self.reputation_factors[policy_name][0, :])

    def getState(self, request_attrs, policy_name):
        arrivalT = request_attrs[1]
        length = request_attrs[2]
        request_service = int(request_attrs[3])
        idleTimes = self.get_oracle_idleT(policy_name)
        reputations = self.get_oracle_reputation(policy_name)
        waitTimes = np.maximum(idleTimes - arrivalT, 0)

        if self.args.State_Mode == "original":
            return np.hstack(([request_service], waitTimes, reputations))

        type_match = np.array([1 if request_service == int(t) else 0 for t in self.oracleTypes])
        req_num = self.reputation_factors[policy_name][0]
        val_num = self.reputation_factors[policy_name][1]
        recent_success_rate = np.divide(val_num, np.maximum(req_num, 1), dtype=float)
        recent_load = req_num / max(self.timeperiodSize, 1)
        # Do not expose true validation probability by default in hard scenarios;
        # otherwise the task becomes too easy and resembles supervised role lookup.
        if getattr(self.args, "Expose_Validation_Prob", False):
            validation_feature = self.oracleValidationProbs
        else:
            validation_feature = np.zeros(self.oracleNum, dtype=float)

        return np.hstack((
            [request_service, length, self.ddl],
            waitTimes,
            reputations,
            self.oracleCost,
            self.oracleAcc,
            type_match,
            validation_feature,
            recent_success_rate,
            recent_load,
        ))

    def getStateP(self, request_id):
        return self.events["BLOR"][3, request_id]

    def _metric_array(self, func):
        values = np.zeros(self.policy_num)
        for i, name in enumerate(self.policy_names):
            values[i] = func(name)
        return values

    def get_accumulateRewards(self, policies, start, end):
        return np.around(self._metric_array(lambda n: np.sum(self.events[n][5, start:end])), 2)

    def get_accumulateCost(self, policies, start, end):
        return np.around(self._metric_array(lambda n: np.sum(self.events[n][8, start:end])), 2)

    def get_FinishTimes(self, policies, start, end):
        return np.around(self._metric_array(lambda n: np.max(self.events[n][4, start:end]) if end > start else 0), 2)

    def get_executeTs(self, policies, start, end):
        return np.around(self._metric_array(lambda n: np.mean(self.events[n][6, start:end]) if end > start else 0), 3)

    def get_waitTs(self, policies, start, end):
        return np.around(self._metric_array(lambda n: np.mean(self.events[n][2, start:end]) if end > start else 0), 3)

    def get_responseTs(self, policies, start, end):
        return np.around(self._metric_array(lambda n: np.mean(self.events[n][3, start:end]) if end > start else 0), 3)

    def get_successTimes(self, policies, start, end):
        denom = max(end - start, 1)
        return np.around(self._metric_array(lambda n: np.sum(self.events[n][7, start:end]) / denom), 3)

    def get_successInTime(self, policies, start, end):
        denom = max(end - start, 1)
        return np.around(self._metric_array(lambda n: np.sum(self.events[n][9, start:end]) / denom), 3)

    def get_rejectTimes(self, policies, start, end):
        return self.get_accumulateCost(policies, start, end)

    def get_totalRewards(self, policies, start):
        return np.around(self._metric_array(lambda n: np.sum(self.events[n][5, start:self.requestNum])), 3)

    def _assigned_count_from_events(self, policy_name, oracle_indices, start=0):
        start = min(max(int(start), 0), self.requestNum - 1)
        chosen = self.events[policy_name][0, start:self.requestNum].astype(int)
        return float(np.sum(np.isin(chosen, oracle_indices)))

    def get_totalMaliciousNum(self, policies, start=0):
        idx = self.malicious_oracles
        return np.around(self._metric_array(lambda n: self._assigned_count_from_events(n, idx, start)), 1)

    def get_totalNormalNum(self, policies, start=0):
        idx = self.normal_oracles
        return np.around(self._metric_array(lambda n: self._assigned_count_from_events(n, idx, start)), 1)

    def get_totalTrustedNum(self, policies, start=0):
        idx = self.trusted_oracles
        return np.around(self._metric_array(lambda n: self._assigned_count_from_events(n, idx, start)), 1)

    def _pb_rate(self, row, start=0):
        start = min(max(int(start), 0), self.requestNum - 1)
        denom = max(self.requestNum - start, 1)
        return np.around(self._metric_array(lambda n: np.sum(self.pb_records[n][row, start:self.requestNum]) / denom), 3)

    def _pb_count(self, row, start=0):
        start = min(max(int(start), 0), self.requestNum - 1)
        return np.around(self._metric_array(lambda n: np.sum(self.pb_records[n][row, start:self.requestNum])), 1)

    def get_totalPrimarySuccessRate(self, policies, start=0):
        return self._pb_rate(0, start)

    def get_totalBackupUsedRate(self, policies, start=0):
        return self._pb_rate(1, start)

    def get_totalBackupRecoveryRate(self, policies, start=0):
        return self._pb_rate(3, start)

    def get_totalBackupSkippedRate(self, policies, start=0):
        return self._pb_rate(10, start)

    def get_totalConditionalBackupRecoveryRate(self, policies, start=0):
        start = min(max(int(start), 0), self.requestNum - 1)
        def _cond(policy_name):
            used = np.sum(self.pb_records[policy_name][1, start:self.requestNum])
            rec = np.sum(self.pb_records[policy_name][3, start:self.requestNum])
            return float(rec / max(used, 1.0))
        return np.around(self._metric_array(_cond), 3)

    def get_totalBackupScoreMean(self, policies, start=0):
        start = min(max(int(start), 0), self.requestNum - 1)
        def _score(policy_name):
            scores = self.pb_records[policy_name][11, start:self.requestNum]
            mask = scores != 0
            return float(np.mean(scores[mask])) if np.any(mask) else 0.0
        return np.around(self._metric_array(_score), 3)

    def get_totalPrimaryMaliciousNum(self, policies, start=0):
        return self._pb_count(6, start)

    def get_totalBackupMaliciousNum(self, policies, start=0):
        return self._pb_count(7, start)

    def get_totalPrimaryTrustedNum(self, policies, start=0):
        return self._pb_count(8, start)

    def get_totalBackupTrustedNum(self, policies, start=0):
        return self._pb_count(9, start)

    def get_totalMatchRate(self, policies, start=0):
        start = min(max(int(start), 0), self.requestNum - 1)
        denom = max(self.requestNum - start, 1)

        def _match_rate(policy_name):
            chosen = self.events[policy_name][0, start:self.requestNum].astype(int)
            chosen_types = self.oracleTypes[chosen]
            request_types = self.request_type[start:self.requestNum]
            return float(np.sum(chosen_types == request_types) / denom)

        return np.around(self._metric_array(_match_rate), 3)

    def get_totalTimes(self, policies, start):
        start_idx = min(max(start, 0), self.requestNum - 1)
        return np.around(self._metric_array(lambda n: np.max(self.events[n][4, :]) - self.arrival_Times[start_idx]), 3)

    def get_all_responseTs(self, policies):
        respTs = np.zeros((self.policy_num, self.requestNum))
        for i, name in enumerate(self.policy_names):
            respTs[i, :] = self.events[name][3, :]
        return np.around(respTs, 3)

    def get_total_responseTs(self, policies, start):
        start = min(max(start, 0), self.requestNum - 1)
        return np.around(self._metric_array(lambda n: np.mean(self.events[n][3, start:self.requestNum])), 3)

    def get_totalSuccess(self, policies, start):
        start = min(max(start, 0), self.requestNum - 1)
        denom = max(self.requestNum - start, 1)
        return np.around(self._metric_array(lambda n: np.sum(self.events[n][7, start:self.requestNum]) / denom), 3)

    def get_totalSuccessInTime(self, policies, start):
        start = min(max(start, 0), self.requestNum - 1)
        denom = max(self.requestNum - start, 1)
        return np.around(self._metric_array(lambda n: np.sum(self.events[n][9, start:self.requestNum]) / denom), 3)

    def get_totalCost(self, policies, start):
        start = min(max(start, 0), self.requestNum - 1)
        denom = max(self.requestNum - start, 1)
        return np.around(self._metric_array(lambda n: np.sum(self.events[n][8, start:self.requestNum]) / denom), 5)
