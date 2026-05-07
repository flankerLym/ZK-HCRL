import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="TCO-DRL scalable oracle selection experiments")

    # General
    parser.add_argument(
        "--Baselines",
        nargs="+",
        default=["Random", "Round-Robin", "Earliest", "DQN", "BLOR", "SemiGreedy", "PPO"],
        help="Methods to compare. Default: Random Round-Robin Earliest DQN BLOR SemiGreedy PPO",
    )
    parser.add_argument("--Baseline_num", type=int, default=0, help="Number of baselines. Set automatically.")
    parser.add_argument("--Epoch", type=int, default=10, help="Training episodes")
    parser.add_argument("--Seed", type=int, default=6, help="Random seed for Python/NumPy/TensorFlow")

    # DQN
    parser.add_argument("--Dqn_start_learn", type=int, default=300, help="Iteration to start DQN learning")
    parser.add_argument("--Dqn_learn_interval", type=int, default=1, help="DQN learning interval")
    parser.add_argument("--Dqn_hidden", type=int, default=96, help="Hidden units for DQN/RA-DDQN. Increase for harder scenarios.")

    parser.add_argument("--Dqn_batch_size", type=int, default=64, help="DQN/RA-DDQN replay minibatch size")
    parser.add_argument("--Dqn_memory_size", type=int, default=3000, help="DQN/RA-DDQN replay memory size")
    parser.add_argument("--Dqn_epsilon_increment", type=float, default=0.0015, help="Exploration-to-exploitation speed for DQN/RA-DDQN")
    parser.add_argument("--Dqn_lr", type=float, default=0.0025, help="Learning rate for NumPy DQN")
    parser.add_argument("--RA_lr", type=float, default=0.0020, help="Learning rate for NumPy RA-DDQN")

    # PPO
    parser.add_argument("--PPO_start_learn", type=int, default=500, help="Iteration to start PPO learning")
    parser.add_argument("--PPO_learn_interval", type=int, default=64, help="PPO update interval")
    parser.add_argument("--PPO_batch_size", type=int, default=64, help="PPO minimum rollout batch size")
    parser.add_argument("--PPO_update_epochs", type=int, default=5, help="PPO update epochs per rollout")
    parser.add_argument("--PPO_hidden", type=int, default=64, help="Hidden units for PPO actor/critic")

    # Optional stronger value-based model
    parser.add_argument("--Use_RA_DDQN", action="store_true", help="Append RA-DDQN as an additional experimental method")
    parser.add_argument("--RA_start_learn", type=int, default=300, help="Iteration to start RA-DDQN learning")
    parser.add_argument("--RA_learn_interval", type=int, default=1, help="RA-DDQN learning interval")

    # Oracle Settings. These defaults are overwritten by the scalable generator below.
    parser.add_argument("--Oracle_Type", type=list, default=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], help="Oracle Type")
    parser.add_argument("--Oracle_Cost", type=list, default=[0.3, 0.3, 0.6, 0.6, 0.9, 0.3, 0.3, 0.6, 0.6, 0.9, 0.3, 0.3, 0.6, 0.6, 0.9], help="Oracle Cost")
    parser.add_argument("--Oracle_Acc", type=list, default=[1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2], help="Oracle processing speed")
    parser.add_argument("--Oracle_Tokens", type=list, default=[150, 150, 300, 300, 500, 150, 150, 300, 300, 500, 150, 150, 300, 300, 500], help="Oracle staked tokens")
    parser.add_argument("--Oracle_Behavior_Probs", type=list, default=[], help="Oracle behavior probabilities")
    parser.add_argument("--Oracle_Validation_Probs", type=list, default=[0.5, 0.8, 0.7, 0.85, 0.95, 0.5, 0.8, 0.7, 0.85, 0.95, 0.5, 0.8, 0.7, 0.85, 0.95], help="Oracle validation probabilities")
    parser.add_argument("--Oracle_Num", type=int, default=15, help="Number of oracles. Automatically set to 3 * Oracles_Per_Type.")
    parser.add_argument("--Oracles_Per_Type", type=int, default=5, help="Number of oracles for each service type. Total Oracle_Num = 3 * Oracles_Per_Type.")
    parser.add_argument("--Service_Type_Num", type=int, default=3, help="Number of request/oracle service types")
    parser.add_argument("--Oracle_Initial_Reputation", type=float, default=0.5, help="Initial reputation of oracles")
    parser.add_argument("--Time_Window_Size", type=int, default=5, help="Reputation sliding time-window size")
    parser.add_argument("--Time_Period_Size", type=int, default=60, help="Number of requests per reputation update period")
    parser.add_argument("--Oracle_capacity", type=int, default=1000, help="Oracle capacity used to convert MI to length")

    # Request Settings
    parser.add_argument("--lamda", type=int, default=5, help="Arrival rate parameter")
    parser.add_argument("--Request_Num", type=int, default=6000, help="Number of requests")
    parser.add_argument("--Request_len_Mean", type=int, default=6000, help="Mean request size")
    parser.add_argument("--Request_len_Std", type=int, default=500, help="Std of request size")
    parser.add_argument("--Request_ddl", type=float, default=7.0, help="Deadline of each request")

    # Extended experiment controls
    parser.add_argument("--Noise_Probability", type=float, default=0.0, help="Probability that an oracle response suffers extra latency noise")
    parser.add_argument("--Noise_Delay", type=float, default=1.0, help="Extra response time added under noise")
    parser.add_argument("--State_Mode", choices=["original", "enhanced"], default="original", help="State representation. original preserves the paper setting; enhanced adds cost/acc/type-match/validation features.")
    parser.add_argument("--Reward_Mode", choices=["original", "risk_aware"], default="original", help="Reward function. original preserves the paper setting; risk_aware adds validation/behavior/timeout penalties.")
    parser.add_argument("--Success_Mode", choices=["original", "validation_aware"], default="original",
                        help="original: success requires deadline+type match; validation_aware: success also requires validation success.")
    parser.add_argument("--Scenario", choices=["static", "validation_stress", "rl_hard", "rl_harder"], default="static",
                        help="static keeps the original-style synthetic environment; validation_stress makes low-cost oracles less reliable; rl_hard adds bursty requests and fatigue traps; rl_harder hides true validation probability and uses stronger non-stationary bait oracles.")
    parser.add_argument("--SemiGreedy_View", choices=["myopic", "risk_aware"], default="myopic",
                        help="myopic keeps SemiGreedy close to the original one-step reward/cost heuristic; risk_aware lets it use expected validation-aware reward.")
    parser.add_argument("--Action_Mask_Mode", choices=["none", "type"], default="none",
                        help="none: normal discrete oracle action space; type: RL agents can only choose oracles matching the request type.")
    parser.add_argument("--Fatigue_Strength", type=float, default=1.0,
                        help="Strength of load-induced validation decay in rl_hard/rl_harder scenarios.")
    parser.add_argument("--Burstiness", type=float, default=0.80,
                        help="Probability of keeping the same request service type in rl_hard/rl_harder scenarios.")
    parser.add_argument("--Expose_Validation_Prob", action="store_true",
                        help="If set, enhanced state includes true oracle validation probabilities. By default, hard scenarios hide this privileged information and use observed history instead.")
    parser.add_argument("--Harder_Request_DDL", type=float, default=6.6,
                        help="Default deadline used by rl_harder when Request_ddl is left at 7.0.")

    # Risk-aware reward weights
    # These weights are used inside a bounded, success-aligned reward.
    # The final per-step reward is clipped by Reward_Clip, so total reward will not explode.
    parser.add_argument("--W_SUCCESS", type=float, default=2.2, help="Direct reward for validation-aware task success: on-time + match + validation")
    parser.add_argument("--W_REPUTATION", type=float, default=0.35)
    parser.add_argument("--W_MATCH", type=float, default=0.8)
    parser.add_argument("--W_VALIDATION", type=float, default=1.2)
    parser.add_argument("--W_COST", type=float, default=1.15)
    parser.add_argument("--W_RESPONSE", type=float, default=1.35)
    parser.add_argument("--W_BEHAVIOR", type=float, default=0.55)
    parser.add_argument("--W_TIMEOUT", type=float, default=2.0)
    parser.add_argument("--Reward_Clip", type=float, default=3.0,
                        help="Clip each risk-aware reward into [-Reward_Clip, Reward_Clip].")
    parser.add_argument("--Reward_Scale", type=float, default=3.0,
                        help="Scale factor applied after reward normalization. Keep small, e.g. 2-4.")

    args = parser.parse_args()

    # Stress scenarios for testing whether learning agents can handle risk beyond
    # one-step cost/type greedy selection.
    if args.Scenario in ["validation_stress", "rl_hard", "rl_harder"]:
        args.Success_Mode = "validation_aware"
        if args.Reward_Mode == "original":
            args.Reward_Mode = "risk_aware"
        if args.State_Mode == "original":
            args.State_Mode = "enhanced"
        # Keep weights moderate. The reward function itself is normalized and clipped.
        # Stress scenarios should optimize validation-aware task success first,
        # then risk/cost as secondary objectives.
        pass

    if args.Scenario in ["rl_hard", "rl_harder"]:
        # Type masking is the action-space innovation: invalid service-type actions
        # are removed for learning agents. The heuristic baselines remain unchanged.
        if args.Action_Mask_Mode == "none":
            args.Action_Mask_Mode = "type"
        # Success-aligned but bounded reward.
        # Do not over-emphasize reputation; otherwise RA-DDQN becomes too conservative.
        if args.W_SUCCESS == 2.2:
            args.W_SUCCESS = 2.6
        if args.W_RESPONSE == 1.35:
            args.W_RESPONSE = 1.5
        if args.W_TIMEOUT == 2.0:
            args.W_TIMEOUT = 2.2

    if args.Scenario == "rl_harder":
        # Harder setting: do not expose true validation probability, make request bursts
        # longer, strengthen fatigue, and slightly tighten deadlines. These changes make
        # one-step greedy selection less reliable and require learning from observed history.
        if args.Burstiness == 0.80:
            args.Burstiness = 0.93
        if args.Fatigue_Strength == 1.0:
            args.Fatigue_Strength = 2.2
        if args.Request_ddl == 7.0:
            args.Request_ddl = args.Harder_Request_DDL
        if args.W_SUCCESS == 2.6:
            args.W_SUCCESS = 3.0
        if args.W_COST == 1.15:
            args.W_COST = 1.25
        if args.W_RESPONSE == 1.5:
            args.W_RESPONSE = 1.7
        if args.W_TIMEOUT == 2.2:
            args.W_TIMEOUT = 2.6
        if args.Dqn_hidden == 96:
            args.Dqn_hidden = 128
        if args.Dqn_start_learn == 300:
            args.Dqn_start_learn = 200
        if args.RA_start_learn == 300:
            args.RA_start_learn = 200

    if args.Use_RA_DDQN and "RA-DDQN" not in args.Baselines:
        args.Baselines.append("RA-DDQN")

    # Auto-generate scalable oracle community.
    # Role pattern matches the original 15-node setup: each service type has malicious, trusted, normal, trusted, trusted.
    role_pattern = ["malicious", "trusted_low", "normal", "trusted_mid", "trusted_high"]

    oracle_type = []
    oracle_cost = []
    oracle_acc = []
    oracle_tokens = []
    oracle_behavior_probs = []
    oracle_validation_probs = []
    oracle_fatigue_sensitivity = []

    malicious_index = []
    normal_index = []
    trusted_index = []

    idx = 0
    for service_type in range(args.Service_Type_Num):
        for k in range(args.Oracles_Per_Type):
            role = role_pattern[k % len(role_pattern)]
            oracle_type.append(service_type)

            if args.Scenario == "validation_stress":
                # Stress setup: the cheapest matching oracle is no longer the safest.
                # This exposes the limitation of myopic greedy selection.
                if role == "malicious":
                    oracle_cost.append(0.25); oracle_acc.append(1.0); oracle_tokens.append(100)
                    oracle_behavior_probs.append([0.40, 0.25, 0.25, 0.10]); oracle_validation_probs.append(0.25)
                    oracle_fatigue_sensitivity.append(0.06); malicious_index.append(idx)
                elif role == "normal":
                    oracle_cost.append(0.45); oracle_acc.append(1.1); oracle_tokens.append(250)
                    oracle_behavior_probs.append([0.65, 0.20, 0.15, 0.00]); oracle_validation_probs.append(0.60)
                    oracle_fatigue_sensitivity.append(0.05); normal_index.append(idx)
                elif role == "trusted_low":
                    oracle_cost.append(0.25); oracle_acc.append(1.0); oracle_tokens.append(150)
                    oracle_behavior_probs.append([0.75, 0.20, 0.05, 0.00]); oracle_validation_probs.append(0.55)
                    oracle_fatigue_sensitivity.append(0.08); trusted_index.append(idx)
                elif role == "trusted_mid":
                    oracle_cost.append(0.65); oracle_acc.append(1.15); oracle_tokens.append(400)
                    oracle_behavior_probs.append([0.90, 0.10, 0.00, 0.00]); oracle_validation_probs.append(0.90)
                    oracle_fatigue_sensitivity.append(0.02); trusted_index.append(idx)
                elif role == "trusted_high":
                    oracle_cost.append(0.95); oracle_acc.append(1.25); oracle_tokens.append(700)
                    oracle_behavior_probs.append([0.98, 0.02, 0.00, 0.00]); oracle_validation_probs.append(0.98)
                    oracle_fatigue_sensitivity.append(0.00); trusted_index.append(idx)
            elif args.Scenario == "rl_hard":
                # RL-hard setup: cheap same-type nodes are bait. They look attractive
                # to one-step greedy because of low cost and high immediate matching,
                # but their validation probability decays quickly when over-used.
                # Robust nodes are more expensive, so a policy must learn a long-term
                # trust-cost trade-off instead of simply picking the cheapest match.
                if role == "malicious":
                    oracle_cost.append(0.18); oracle_acc.append(1.20); oracle_tokens.append(80)
                    oracle_behavior_probs.append([0.25, 0.25, 0.30, 0.20]); oracle_validation_probs.append(0.28)
                    oracle_fatigue_sensitivity.append(0.12); malicious_index.append(idx)
                elif role == "trusted_low":
                    oracle_cost.append(0.22); oracle_acc.append(1.20); oracle_tokens.append(160)
                    oracle_behavior_probs.append([0.70, 0.20, 0.10, 0.00]); oracle_validation_probs.append(0.72)
                    oracle_fatigue_sensitivity.append(0.18); trusted_index.append(idx)
                elif role == "normal":
                    oracle_cost.append(0.42); oracle_acc.append(1.08); oracle_tokens.append(260)
                    oracle_behavior_probs.append([0.62, 0.22, 0.16, 0.00]); oracle_validation_probs.append(0.62)
                    oracle_fatigue_sensitivity.append(0.10); normal_index.append(idx)
                elif role == "trusted_mid":
                    oracle_cost.append(0.68); oracle_acc.append(1.05); oracle_tokens.append(480)
                    oracle_behavior_probs.append([0.94, 0.06, 0.00, 0.00]); oracle_validation_probs.append(0.93)
                    oracle_fatigue_sensitivity.append(0.02); trusted_index.append(idx)
                elif role == "trusted_high":
                    oracle_cost.append(0.98); oracle_acc.append(1.15); oracle_tokens.append(800)
                    oracle_behavior_probs.append([0.99, 0.01, 0.00, 0.00]); oracle_validation_probs.append(0.99)
                    oracle_fatigue_sensitivity.append(0.00); trusted_index.append(idx)
            elif args.Scenario == "rl_harder":
                # Harder setup: the bait nodes have high initial validation and low cost,
                # so a short run can be misleading. Under bursty traffic they fatigue
                # sharply. The model must infer this from recent success/load; true
                # validation probabilities are hidden from state by default.
                if role == "malicious":
                    oracle_cost.append(0.16); oracle_acc.append(1.25); oracle_tokens.append(90)
                    oracle_behavior_probs.append([0.45, 0.25, 0.20, 0.10]); oracle_validation_probs.append(0.62)
                    oracle_fatigue_sensitivity.append(0.30); malicious_index.append(idx)
                elif role == "trusted_low":
                    oracle_cost.append(0.20); oracle_acc.append(1.25); oracle_tokens.append(180)
                    oracle_behavior_probs.append([0.78, 0.17, 0.05, 0.00]); oracle_validation_probs.append(0.84)
                    oracle_fatigue_sensitivity.append(0.42); trusted_index.append(idx)
                elif role == "normal":
                    oracle_cost.append(0.38); oracle_acc.append(1.12); oracle_tokens.append(280)
                    oracle_behavior_probs.append([0.66, 0.21, 0.13, 0.00]); oracle_validation_probs.append(0.72)
                    oracle_fatigue_sensitivity.append(0.24); normal_index.append(idx)
                elif role == "trusted_mid":
                    oracle_cost.append(0.72); oracle_acc.append(1.05); oracle_tokens.append(520)
                    oracle_behavior_probs.append([0.93, 0.07, 0.00, 0.00]); oracle_validation_probs.append(0.91)
                    oracle_fatigue_sensitivity.append(0.08); trusted_index.append(idx)
                elif role == "trusted_high":
                    oracle_cost.append(1.08); oracle_acc.append(1.12); oracle_tokens.append(900)
                    oracle_behavior_probs.append([0.985, 0.015, 0.00, 0.00]); oracle_validation_probs.append(0.97)
                    oracle_fatigue_sensitivity.append(0.02); trusted_index.append(idx)
            else:
                if role == "malicious":
                    oracle_cost.append(0.3)
                    oracle_acc.append(1.0)
                    oracle_tokens.append(150)
                    oracle_behavior_probs.append([0.5, 0.25, 0.2, 0.05])
                    oracle_validation_probs.append(0.5)
                    malicious_index.append(idx)
                elif role == "normal":
                    oracle_cost.append(0.6)
                    oracle_acc.append(1.1)
                    oracle_tokens.append(300)
                    oracle_behavior_probs.append([0.65, 0.2, 0.15, 0.0])
                    oracle_validation_probs.append(0.7)
                    normal_index.append(idx)
                elif role == "trusted_low":
                    oracle_cost.append(0.3)
                    oracle_acc.append(1.0)
                    oracle_tokens.append(150)
                    oracle_behavior_probs.append([0.9, 0.1, 0.0, 0.0])
                    oracle_validation_probs.append(0.8)
                    trusted_index.append(idx)
                elif role == "trusted_mid":
                    oracle_cost.append(0.6)
                    oracle_acc.append(1.1)
                    oracle_tokens.append(300)
                    oracle_behavior_probs.append([0.9, 0.1, 0.0, 0.0])
                    oracle_validation_probs.append(0.85)
                    trusted_index.append(idx)
                elif role == "trusted_high":
                    oracle_cost.append(0.9)
                    oracle_acc.append(1.2)
                    oracle_tokens.append(500)
                    oracle_behavior_probs.append([0.9, 0.1, 0.0, 0.0])
                    oracle_validation_probs.append(0.95)
                    trusted_index.append(idx)
            idx += 1

    args.Oracle_Num = args.Service_Type_Num * args.Oracles_Per_Type
    args.Oracle_Type = oracle_type
    args.Oracle_Cost = oracle_cost
    args.Oracle_Acc = oracle_acc
    args.Oracle_Tokens = oracle_tokens
    args.Oracle_Behavior_Probs = oracle_behavior_probs
    args.Oracle_Validation_Probs = oracle_validation_probs
    if len(oracle_fatigue_sensitivity) < len(oracle_type):
        oracle_fatigue_sensitivity.extend([0.0] * (len(oracle_type) - len(oracle_fatigue_sensitivity)))
    args.Oracle_Fatigue_Sensitivity = oracle_fatigue_sensitivity
    args.Malicious_Oracle_Index = malicious_index
    args.Normal_Oracle_Index = normal_index
    args.Trusted_Oracle_Index = trusted_index
    args.Baseline_num = len(args.Baselines)

    print("Generated Oracle_Num:", args.Oracle_Num)
    print("Scenario:", args.Scenario, "Success_Mode:", args.Success_Mode, "State_Mode:", args.State_Mode, "Reward_Mode:", args.Reward_Mode, "Action_Mask_Mode:", args.Action_Mask_Mode)
    print("Baselines:", args.Baselines)
    print("Malicious oracles:", args.Malicious_Oracle_Index)
    print("Normal oracles:", args.Normal_Oracle_Index)
    print("Trusted oracles:", args.Trusted_Oracle_Index)
    return args
