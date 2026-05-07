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
    parser.add_argument("--Epoch", type=int, default=5, help="Training episodes")
    parser.add_argument("--Seed", type=int, default=6, help="Random seed for Python/NumPy/TensorFlow")

    # DQN
    parser.add_argument("--Dqn_start_learn", type=int, default=500, help="Iteration to start DQN learning")
    parser.add_argument("--Dqn_learn_interval", type=int, default=1, help="DQN learning interval")
    parser.add_argument("--Dqn_hidden", type=int, default=64, help="Hidden units for DQN. Increase for many oracles.")

    # PPO
    parser.add_argument("--PPO_start_learn", type=int, default=500, help="Iteration to start PPO learning")
    parser.add_argument("--PPO_learn_interval", type=int, default=64, help="PPO update interval")
    parser.add_argument("--PPO_batch_size", type=int, default=64, help="PPO minimum rollout batch size")
    parser.add_argument("--PPO_update_epochs", type=int, default=5, help="PPO update epochs per rollout")
    parser.add_argument("--PPO_hidden", type=int, default=64, help="Hidden units for PPO actor/critic")

    # Optional stronger value-based model
    parser.add_argument("--Use_RA_DDQN", action="store_true", help="Append RA-DDQN as an additional experimental method")
    parser.add_argument("--RA_start_learn", type=int, default=500, help="Iteration to start RA-DDQN learning")
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
    parser.add_argument("--Scenario", choices=["static", "validation_stress"], default="static",
                        help="static keeps the original-style synthetic environment; validation_stress makes low-cost oracles less reliable and evaluates validation-aware success.")
    parser.add_argument("--SemiGreedy_View", choices=["myopic", "risk_aware"], default="myopic",
                        help="myopic keeps SemiGreedy close to the original one-step reward/cost heuristic; risk_aware lets it use expected validation-aware reward.")

    # Risk-aware reward weights
    parser.add_argument("--W_REPUTATION", type=float, default=1.0)
    parser.add_argument("--W_MATCH", type=float, default=2.0)
    parser.add_argument("--W_VALIDATION", type=float, default=1.0)
    parser.add_argument("--W_COST", type=float, default=1.0)
    parser.add_argument("--W_RESPONSE", type=float, default=0.5)
    parser.add_argument("--W_BEHAVIOR", type=float, default=2.0)
    parser.add_argument("--W_TIMEOUT", type=float, default=2.0)

    args = parser.parse_args()

    # A harder, realistic stress scenario: cheap oracles can be less reliable, so
    # methods that greedily optimize immediate cost/type matching no longer dominate.
    # This does not disable SemiGreedy; it changes the environment objective to
    # validation-aware oracle service quality.
    if args.Scenario == "validation_stress":
        args.Success_Mode = "validation_aware"
        if args.Reward_Mode == "original":
            args.Reward_Mode = "risk_aware"
        if args.State_Mode == "original":
            args.State_Mode = "enhanced"
        # Make validation feedback matter enough for learning-based methods.
        if args.W_VALIDATION == 1.0:
            args.W_VALIDATION = 3.0
        if args.W_COST == 1.0:
            args.W_COST = 0.6
        if args.W_BEHAVIOR == 2.0:
            args.W_BEHAVIOR = 1.5

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
                # This is meant to expose the limitation of myopic greedy selection.
                if role == "malicious":
                    oracle_cost.append(0.25)
                    oracle_acc.append(1.0)
                    oracle_tokens.append(100)
                    oracle_behavior_probs.append([0.40, 0.25, 0.25, 0.10])
                    oracle_validation_probs.append(0.25)
                    malicious_index.append(idx)
                elif role == "normal":
                    oracle_cost.append(0.45)
                    oracle_acc.append(1.1)
                    oracle_tokens.append(250)
                    oracle_behavior_probs.append([0.65, 0.20, 0.15, 0.00])
                    oracle_validation_probs.append(0.60)
                    normal_index.append(idx)
                elif role == "trusted_low":
                    oracle_cost.append(0.25)
                    oracle_acc.append(1.0)
                    oracle_tokens.append(150)
                    oracle_behavior_probs.append([0.75, 0.20, 0.05, 0.00])
                    oracle_validation_probs.append(0.55)
                    trusted_index.append(idx)
                elif role == "trusted_mid":
                    oracle_cost.append(0.65)
                    oracle_acc.append(1.15)
                    oracle_tokens.append(400)
                    oracle_behavior_probs.append([0.90, 0.10, 0.00, 0.00])
                    oracle_validation_probs.append(0.90)
                    trusted_index.append(idx)
                elif role == "trusted_high":
                    oracle_cost.append(0.95)
                    oracle_acc.append(1.25)
                    oracle_tokens.append(700)
                    oracle_behavior_probs.append([0.98, 0.02, 0.00, 0.00])
                    oracle_validation_probs.append(0.98)
                    trusted_index.append(idx)
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
    args.Malicious_Oracle_Index = malicious_index
    args.Normal_Oracle_Index = normal_index
    args.Trusted_Oracle_Index = trusted_index
    args.Baseline_num = len(args.Baselines)

    print("Generated Oracle_Num:", args.Oracle_Num)
    print("Scenario:", args.Scenario, "Success_Mode:", args.Success_Mode, "State_Mode:", args.State_Mode, "Reward_Mode:", args.Reward_Mode)
    print("Baselines:", args.Baselines)
    print("Malicious oracles:", args.Malicious_Oracle_Index)
    print("Normal oracles:", args.Normal_Oracle_Index)
    print("Trusted oracles:", args.Trusted_Oracle_Index)
    return args
