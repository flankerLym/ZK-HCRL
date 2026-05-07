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
            # request type, request length, ddl + wait + reputation + cost + acc + type_match + validation_prob
            self.s_features = 3 + 6 * self.oracleNum

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
            self.s_features = 3 + 6 * self.oracleNum

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
        args = self.args
        timeout = 1 if durationT > ddl else 0
        behavior_risk = np.log1p(behavior_record) / np.log1p(100)
        normalized_response = durationT / max(ddl, 1e-8)
        return (
            args.W_REPUTATION * reputation
            + args.W_MATCH * match
            + args.W_VALIDATION * successful_validation
            - args.W_COST * cost
            - args.W_RESPONSE * normalized_response
            - args.W_BEHAVIOR * behavior_risk
            - args.W_TIMEOUT * timeout
        )

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
        validation_prob = self.oracleValidationProbs[action]
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
                expected_validation = self.oracleValidationProbs[action] if durationT <= request_attrs[4] else 0.0
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
        return np.hstack((
            [request_service, length, self.ddl],
            waitTimes,
            reputations,
            self.oracleCost,
            self.oracleAcc,
            type_match,
            self.oracleValidationProbs,
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

    def get_totalMaliciousNum(self, policies):
        idx = self.malicious_oracles
        return np.around(self._metric_array(lambda n: np.sum(self.oracle_events[n][1, idx])), 1)

    def get_totalNormalNum(self, policies):
        idx = self.normal_oracles
        return np.around(self._metric_array(lambda n: np.sum(self.oracle_events[n][1, idx])), 1)

    def get_totalTrustedNum(self, policies):
        idx = self.trusted_oracles
        return np.around(self._metric_array(lambda n: np.sum(self.oracle_events[n][1, idx])), 1)

    def get_totalMatchRate(self, policies):
        return np.around(self._metric_array(lambda n: np.sum(self.oracle_events[n][3, :]) / max(self.requestNum, 1)), 3)

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
