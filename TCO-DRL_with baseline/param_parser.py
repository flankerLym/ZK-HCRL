import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="SAIRL")

    # General
    parser.add_argument("--Baselines",
                        type=list,
                        default=['Random', 'Round-Robin', 'Earliest', 'DQN', 'BLOR', 'SemiGreedy'],
                        help="Experiment Baseline")
    parser.add_argument("--Baseline_num",
                        type=int,
                        default=0,
                        help="Number of baselines")


    parser.add_argument("--Epoch",
                        type=int,
                        default=5,
                        help="Training Epochs")

    # DQN
    parser.add_argument("--Dqn_start_learn",
                        type=int,
                        default=500,
                        help="Iteration start Learn for normal dqn")
    parser.add_argument("--Dqn_learn_interval",
                        type=int,
                        default=1,
                        help="Dqn's learning interval")

    # Oracle Settings
    parser.add_argument("--Oracle_Type",
                        type=list,
                        default=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                        help="Oracle Type")
    parser.add_argument("--Oracle_Cost",
                        type=list,
                        default=[0.3, 0.3, 0.6, 0.6, 0.9, 0.3, 0.3, 0.6, 0.6, 0.9, 0.3, 0.3, 0.6, 0.6, 0.9],
                        help="Oracle Cost")
    parser.add_argument("--Oracle_Acc",
                        type=list,
                        default=[1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2],
                        help="Oracle Cpus")
    parser.add_argument("--Oracle_Tokens",
                        type=list,
                        default=[150, 150, 300, 300, 500, 150, 150, 300, 300, 500, 150, 150, 300, 300, 500],
                        help="Oracle Staked tokens")
    parser.add_argument("--Oracle_Behavior_Probs",
                        type=list,
                        default=[[0.5, 0.25, 0.2, 0.05], [0.9, 0.1, 0, 0], [0.65, 0.2, 0.15, 0], [0.9, 0.1, 0, 0], [0.9, 0.1, 0, 0],
                                 [0.5, 0.25, 0.2, 0.05], [0.9, 0.1, 0, 0], [0.65, 0.2, 0.15, 0], [0.9, 0.1, 0, 0], [0.9, 0.1, 0, 0],
                                 [0.5, 0.25, 0.2, 0.05], [0.9, 0.1, 0, 0], [0.65, 0.2, 0.15, 0], [0.9, 0.1, 0, 0], [0.9, 0.1, 0, 0]],

                        help="Oracle Behavior Probs")
    parser.add_argument("--Oracle_Validation_Probs",
                        type=list,
                        default=[0.5, 0.8, 0.7, 0.85, 0.95, 0.5, 0.8, 0.7, 0.85, 0.95, 0.5, 0.8, 0.7, 0.85, 0.95],

                        help="Oracle Validation Probs")
    parser.add_argument("--Oracle_Num",
                        type=int,
                        default=15,
                        help="The number of Oracles")
    parser.add_argument("--Oracle_Initial_Reputation",
                        type=float,
                        default=0.5,
                        help="The initial reputation of Oracles")
    parser.add_argument("--Time_Window_Size",
                        type=int,
                        default=5,
                        help="The size of Time window")
    parser.add_argument("--Time_Period_Size",
                        type=int,
                        default=60,
                        help="The size of Time period")
    parser.add_argument("--Oracle_capacity",
                        type=int,
                        default=1000,
                        help="Oracle capacity")

    # Request Settings
    parser.add_argument("--lamda",
                        type=int,
                        default=5,
                        help="The parameter used to control the interval time of each requests.")
    parser.add_argument("--Request_Num",
                        type=int,
                        default=6000,
                        help="The number of requests.")
    parser.add_argument("--Request_len_Mean",
                        type=int,
                        default=6000,
                        help="The mean value of the normal distribution.")
    parser.add_argument("--Request_len_Std",
                        type=int,
                        default=500,
                        help="The std value of the normal distribution.")
    parser.add_argument("--Request_ddl",
                        type=float,
                        default=7.0,
                        help="Deadline time of each requests")

    parser.add_argument("--Oracles_Per_Type",
                        type=int,
                        default=5,
                        help="Number of oracles for each service type. Total Oracle_Num = 3 * Oracles_Per_Type.")

    args = parser.parse_args()

    # 自动生成 Oracle 参数
    # 原始每 5 个 oracle 的角色模式：
    # 0: malicious, 1: trusted, 2: normal, 3: trusted, 4: trusted
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

    service_type_num = 3
    idx = 0

    for service_type in range(service_type_num):
        for k in range(args.Oracles_Per_Type):
            role = role_pattern[k % len(role_pattern)]

            oracle_type.append(service_type)

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
                oracle_behavior_probs.append([0.65, 0.2, 0.15, 0])
                oracle_validation_probs.append(0.7)
                normal_index.append(idx)

            elif role == "trusted_low":
                oracle_cost.append(0.3)
                oracle_acc.append(1.0)
                oracle_tokens.append(150)
                oracle_behavior_probs.append([0.9, 0.1, 0, 0])
                oracle_validation_probs.append(0.8)
                trusted_index.append(idx)

            elif role == "trusted_mid":
                oracle_cost.append(0.6)
                oracle_acc.append(1.1)
                oracle_tokens.append(300)
                oracle_behavior_probs.append([0.9, 0.1, 0, 0])
                oracle_validation_probs.append(0.85)
                trusted_index.append(idx)

            elif role == "trusted_high":
                oracle_cost.append(0.9)
                oracle_acc.append(1.2)
                oracle_tokens.append(500)
                oracle_behavior_probs.append([0.9, 0.1, 0, 0])
                oracle_validation_probs.append(0.95)
                trusted_index.append(idx)

            idx += 1

    args.Oracle_Num = service_type_num * args.Oracles_Per_Type
    args.Oracle_Type = oracle_type
    args.Oracle_Cost = oracle_cost
    args.Oracle_Acc = oracle_acc
    args.Oracle_Tokens = oracle_tokens
    args.Oracle_Behavior_Probs = oracle_behavior_probs
    args.Oracle_Validation_Probs = oracle_validation_probs

    args.Malicious_Oracle_Index = malicious_index
    args.Normal_Oracle_Index = normal_index
    args.Trusted_Oracle_Index = trusted_index

    print("Generated Oracle_Num:", args.Oracle_Num)
    print("Malicious oracles:", args.Malicious_Oracle_Index)
    print("Normal oracles:", args.Normal_Oracle_Index)
    print("Trusted oracles:", args.Trusted_Oracle_Index)

    return args