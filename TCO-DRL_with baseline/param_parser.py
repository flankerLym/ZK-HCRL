import argparse
import numpy as np

BASE_METHODS = ["Random", "Round-Robin", "Earliest", "DQN", "BLOR", "SemiGreedy", "PPO"]
OPTIONAL_METHODS = ["RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle"]
ALL_METHODS = BASE_METHODS + OPTIONAL_METHODS

METHOD_ALIASES = {
    "random": "Random",
    "rr": "Round-Robin", "round-robin": "Round-Robin", "round_robin": "Round-Robin",
    "earliest": "Earliest", "early": "Earliest",
    "dqn": "DQN", "tco-drl": "DQN", "tco_drl": "DQN",
    "blor": "BLOR",
    "semigreedy": "SemiGreedy", "semi-greedy": "SemiGreedy", "semi_greedy": "SemiGreedy", "psg": "SemiGreedy",
    "ppo": "PPO",
    "ra": "RA-DDQN", "ra-ddqn": "RA-DDQN", "ra_ddqn": "RA-DDQN",
    "pb": "PB-SafeDQN", "pb-safe": "PB-SafeDQN", "pb-safedqn": "PB-SafeDQN", "pb_safedqn": "PB-SafeDQN",
    "cobra": "COBRA-Oracle", "cobra-oracle": "COBRA-Oracle", "cobra_oracle": "COBRA-Oracle",
    "hcrl": "HCRL-Oracle", "hcrl-oracle": "HCRL-Oracle", "hcrl_oracle": "HCRL-Oracle",
}

METHOD_PRESETS = {
    "base": BASE_METHODS,
    "default": BASE_METHODS,
    "all": ALL_METHODS,
    "paper_all": ALL_METHODS,
    "rl_only": ["DQN", "PPO", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle"],
    "fast": ["DQN", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle"],
    "cobra": ["DQN", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle"],
    "hcrl": ["DQN", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle"],
    "hcrl_only": ["HCRL-Oracle"],
    "pb_only": ["PB-SafeDQN"],
    "cobra_only": ["COBRA-Oracle"],
    "ra_only": ["RA-DDQN"],
    "dqn_only": ["DQN"],
}


def _canonical_method(name):
    key = str(name).strip()
    if key in ALL_METHODS:
        return key
    k1 = key.lower().replace(" ", "")
    if k1 in METHOD_ALIASES:
        return METHOD_ALIASES[k1]
    k2 = key.lower()
    if k2 in METHOD_ALIASES:
        return METHOD_ALIASES[k2]
    valid = ", ".join(ALL_METHODS + sorted(METHOD_ALIASES.keys()))
    raise ValueError(f"Unknown method '{name}'. Valid methods/aliases: {valid}")


def _dedupe(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def _resolve_selected_methods(args):
    if getattr(args, "Methods", None):
        methods = [_canonical_method(m) for m in args.Methods]
        mode = "exact"
    elif getattr(args, "Method_Preset", None):
        methods = list(METHOD_PRESETS[args.Method_Preset])
        mode = f"preset:{args.Method_Preset}"
    else:
        methods = list(args.Baselines if args.Baselines is not None else BASE_METHODS)
        methods = [_canonical_method(m) for m in methods]
        mode = "legacy"
        for flag, m in [
            (args.Use_RA_DDQN, "RA-DDQN"),
            (args.Use_PB_SafeDQN, "PB-SafeDQN"),
            (args.Use_COBRA, "COBRA-Oracle"),
            (args.Use_HCRL, "HCRL-Oracle"),
        ]:
            if flag and m not in methods:
                methods.append(m)
    methods = _dedupe(methods)
    if not methods:
        raise ValueError("No methods selected.")
    args.Baselines = methods
    args.Baseline_num = len(methods)
    args.Method_Selection_Mode = mode
    args.Use_RA_DDQN = "RA-DDQN" in methods
    args.Use_PB_SafeDQN = "PB-SafeDQN" in methods
    args.Use_COBRA = "COBRA-Oracle" in methods
    args.Use_HCRL = "HCRL-Oracle" in methods
    return args


def parameter_parser():
    p = argparse.ArgumentParser(description="TCO-DRL / Audit-aware HCRL-Oracle experiments")

    p.add_argument("--Baselines", nargs="+", default=None)
    p.add_argument("--Methods", "--Run_Methods", nargs="+", default=None,
                   help="Exact methods to run, e.g. --Methods DQN HCRL")
    p.add_argument("--Method_Preset", choices=sorted(METHOD_PRESETS.keys()), default=None)
    p.add_argument("--List_Methods", action="store_true")
    p.add_argument("--Baseline_num", type=int, default=0)
    p.add_argument("--Epoch", type=int, default=10)
    p.add_argument("--Seed", type=int, default=6)
    p.add_argument("--Output_Dir", type=str, default="output")
    p.add_argument("--Run_Tag", type=str, default="")

    # Learning models.
    p.add_argument("--Dqn_start_learn", type=int, default=300)
    p.add_argument("--Dqn_learn_interval", type=int, default=4)
    p.add_argument("--Dqn_hidden", type=int, default=96)
    p.add_argument("--Dqn_batch_size", type=int, default=64)
    p.add_argument("--Dqn_memory_size", type=int, default=3000)
    p.add_argument("--Dqn_epsilon_increment", type=float, default=0.0015)
    p.add_argument("--Dqn_lr", type=float, default=0.0025)

    p.add_argument("--PPO_start_learn", type=int, default=500)
    p.add_argument("--PPO_learn_interval", type=int, default=64)
    p.add_argument("--PPO_batch_size", type=int, default=64)
    p.add_argument("--PPO_update_epochs", type=int, default=4)
    p.add_argument("--PPO_hidden", type=int, default=64)
    p.add_argument("--PPO_lr", type=float, default=0.0015)

    p.add_argument("--Use_RA_DDQN", action="store_true")
    p.add_argument("--RA_lr", type=float, default=0.0020)
    p.add_argument("--RA_start_learn", type=int, default=300)
    p.add_argument("--RA_learn_interval", type=int, default=4)

    p.add_argument("--Use_PB_SafeDQN", action="store_true")
    p.add_argument("--PB_lr", type=float, default=0.0022)
    p.add_argument("--PB_start_learn", type=int, default=200)
    p.add_argument("--PB_learn_interval", type=int, default=4)
    p.add_argument("--PB_Backup_Mode", choices=["parallel", "serial"], default="parallel")
    p.add_argument("--PB_Backup_Trigger", choices=["always", "cost_aware"], default="cost_aware")
    p.add_argument("--PB_Min_Backup_Score", type=float, default=0.38)
    p.add_argument("--PB_Backup_Recovery_Bonus", type=float, default=0.38)
    p.add_argument("--PB_Backup_Used_Penalty", type=float, default=0.16)
    p.add_argument("--PB_Primary_Success_Bonus", type=float, default=0.18)
    p.add_argument("--PB_Backup_Skip_Penalty", type=float, default=0.04)
    p.add_argument("--PB_Backup_Cost_Limit", type=float, default=1.05)
    p.add_argument("--PB_W_RECENT_SUCCESS", type=float, default=0.42)
    p.add_argument("--PB_W_REPUTATION", type=float, default=0.24)
    p.add_argument("--PB_W_LOAD", type=float, default=0.18)
    p.add_argument("--PB_W_COST", type=float, default=0.10)
    p.add_argument("--PB_W_TOKEN", type=float, default=0.14)
    p.add_argument("--PB_W_BEHAVIOR_RISK", type=float, default=0.20)
    p.add_argument("--PB_W_DELAY", type=float, default=0.10)
    p.add_argument("--PB_Prior_Strength", type=float, default=2.0)

    # COBRA-Oracle.
    p.add_argument("--Use_COBRA", action="store_true")
    p.add_argument("--COBRA_lr", type=float, default=0.0016)
    p.add_argument("--COBRA_start_learn", type=int, default=200)
    p.add_argument("--COBRA_learn_interval", type=int, default=4)
    p.add_argument("--COBRA_Teacher_Source", choices=["DQN", "RA-DDQN", "none"], default="DQN")
    p.add_argument("--COBRA_WarmStart_Episode", type=int, default=3)
    p.add_argument("--COBRA_Teacher_Guidance_Episodes", type=int, default=8)
    p.add_argument("--COBRA_Teacher_Start_Prob", type=float, default=0.75)
    p.add_argument("--COBRA_Min_Teacher_Prob", type=float, default=0.05)
    p.add_argument("--COBRA_Gate_Mode", choices=["adaptive", "fixed", "always", "never"], default="adaptive")
    p.add_argument("--COBRA_Min_Backup_Score", type=float, default=0.46)
    p.add_argument("--COBRA_Gate_Alpha", type=float, default=0.15)
    p.add_argument("--COBRA_Gate_Window", type=int, default=400)
    p.add_argument("--COBRA_Primary_Success_Bonus", type=float, default=0.26)
    p.add_argument("--COBRA_Backup_Recovery_Bonus", type=float, default=0.34)
    p.add_argument("--COBRA_Backup_Used_Penalty", type=float, default=0.22)
    p.add_argument("--COBRA_Backup_Skip_Penalty", type=float, default=0.03)
    p.add_argument("--COBRA_Cost_Budget", type=float, default=1.00)
    p.add_argument("--COBRA_Latency_Budget", type=float, default=6.0)
    p.add_argument("--COBRA_Risk_Budget", type=float, default=0.08)
    p.add_argument("--COBRA_Lambda_Cost", type=float, default=0.45)
    p.add_argument("--COBRA_Lambda_Latency", type=float, default=0.35)
    p.add_argument("--COBRA_Lambda_Risk", type=float, default=0.65)
    p.add_argument("--COBRA_Primary_Malicious_Penalty", type=float, default=0.30)
    p.add_argument("--COBRA_Random_Backup", action="store_true")
    p.add_argument("--COBRA_No_Teacher", action="store_true")
    p.add_argument("--COBRA_No_Decoupled_Reward", action="store_true")

    # HCRL-Oracle: risk-aware five-mode hierarchy.
    p.add_argument("--Use_HCRL", action="store_true")
    p.add_argument("--HCRL_lr", type=float, default=0.0013)
    p.add_argument("--HCRL_Mode_lr", type=float, default=0.0010)
    p.add_argument("--HCRL_Use_Actor_Critic", action="store_true", default=True)
    p.add_argument("--HCRL_AC_Entropy", type=float, default=0.05)
    p.add_argument("--HCRL_AC_Value_Coef", type=float, default=0.5)
    p.add_argument("--HCRL_start_learn", type=int, default=200)
    p.add_argument("--HCRL_learn_interval", type=int, default=8)
    p.add_argument("--HCRL_Backup_learn_interval", type=int, default=8)
    p.add_argument("--HCRL_Mode_learn_interval", type=int, default=8)
    p.add_argument("--HCRL_Teacher_Source", choices=["DQN", "RA-DDQN", "COBRA-Oracle", "none"], default="DQN")
    p.add_argument("--HCRL_WarmStart_Episode", type=int, default=3)
    p.add_argument("--HCRL_Teacher_Guidance_Episodes", type=int, default=20)
    p.add_argument("--HCRL_Teacher_Start_Prob", type=float, default=0.85)
    p.add_argument("--HCRL_Min_Teacher_Prob", type=float, default=0.03)
    p.add_argument("--HCRL_Mode_Start_Prob", type=float, default=0.0)
    p.add_argument("--HCRL_Mode_Min_Prob", type=float, default=0.0)
    p.add_argument("--HCRL_Primary_Success_Bonus", type=float, default=0.30)
    p.add_argument("--HCRL_Backup_Recovery_Bonus", type=float, default=0.72)
    p.add_argument("--HCRL_Backup_Used_Penalty", type=float, default=0.08)
    p.add_argument("--HCRL_Unnecessary_Backup_Penalty", type=float, default=0.18)
    p.add_argument("--HCRL_Skip_Recovery_Penalty", type=float, default=0.20)
    p.add_argument("--HCRL_Primary_Malicious_Penalty", type=float, default=0.80)
    p.add_argument("--HCRL_Backup_Malicious_Penalty", type=float, default=1.20)
    p.add_argument("--HCRL_Backup_Guidance_Episodes", type=int, default=20)
    p.add_argument("--HCRL_Backup_Start_Prob", type=float, default=0.95)
    p.add_argument("--HCRL_Backup_Min_Prob", type=float, default=0.05)
    p.add_argument("--HCRL_Cost_Budget", type=float, default=1.00)
    p.add_argument("--HCRL_Latency_Budget", type=float, default=6.20)
    p.add_argument("--HCRL_Risk_Budget", type=float, default=0.06)
    p.add_argument("--HCRL_Lambda_Cost", type=float, default=0.70)
    p.add_argument("--HCRL_Lambda_Latency", type=float, default=0.40)
    p.add_argument("--HCRL_Lambda_Risk", type=float, default=1.20)
    p.add_argument("--HCRL_Primal_Dual", action="store_true", default=True)
    p.add_argument("--HCRL_Lambda_LR", type=float, default=0.008)
    p.add_argument("--HCRL_Lambda_Min", type=float, default=0.0)
    p.add_argument("--HCRL_Lambda_Max", type=float, default=3.0)
    p.add_argument("--HCRL_Parallel_Cost_Discount", type=float, default=0.85)
    p.add_argument("--HCRL_Mode_Names", nargs="+",
                   default=["single_cost", "single_safe", "serial_safe", "parallel_fast", "parallel_safe"],
                   help="Risk-aware high-level modes for HCRL.")
    p.add_argument("--HCRL_No_Safety_Gate", action="store_true")
    p.add_argument("--HCRL_No_Risk_Budgeted_Gate", action="store_true")
    p.add_argument("--HCRL_Safety_Recovery_Mode", choices=["auto", "serial", "parallel"], default="auto")
    p.add_argument("--HCRL_Safety_Min_Backup_Score", type=float, default=0.12)
    p.add_argument("--HCRL_Safety_Primary_Risk_Threshold", type=float, default=0.52)
    p.add_argument("--HCRL_Safety_Score_Margin", type=float, default=0.05)
    p.add_argument("--HCRL_Final_Success_Bonus", type=float, default=0.35)
    p.add_argument("--HCRL_Success_Gain_Bonus", type=float, default=0.45)
    p.add_argument("--HCRL_Safety_Override_Bonus", type=float, default=0.12)
    p.add_argument("--HCRL_Backup_Max_Estimated_Risk", type=float, default=0.42)
    p.add_argument("--HCRL_Backup_Risk_Margin", type=float, default=0.08)
    p.add_argument("--HCRL_Backup_Cost_Cap", type=float, default=1.05)
    p.add_argument("--HCRL_Recovery_Cost_Hard_Cap", type=float, default=1.30)
    p.add_argument("--HCRL_Estimated_Risk_Penalty", type=float, default=0.45)
    p.add_argument("--HCRL_Total_Cost_Penalty", type=float, default=0.18)
    p.add_argument("--HCRL_Trusted_Selection_Bonus", type=float, default=0.12)
    p.add_argument("--HCRL_Backup_Trust_Bonus", type=float, default=0.15)
    p.add_argument("--HCRL_No_Teacher", action="store_true")
    p.add_argument("--HCRL_No_Constrained", action="store_true")
    p.add_argument("--HCRL_No_Decoupled_Reward", action="store_true")
    p.add_argument("--HCRL_Random_Backup", action="store_true")
    p.add_argument("--HCRL_Fixed_Single_Mode", action="store_true")
    p.add_argument("--HCRL_Fixed_Parallel_Mode", action="store_true")

    # Audit-aware reputation adjustment.
    p.add_argument("--Use_Audit_Reputation", action="store_true", default=True,
                   help="Enable hidden/risk-triggered oracle audit and reputation correction. Enabled by default.")
    p.add_argument("--Disable_Audit_Reputation", action="store_true",
                   help="Ablation: disable audit-adjusted reputation.")
    p.add_argument("--Audit_Base_Rate", type=float, default=0.03)
    p.add_argument("--Audit_Risk_Rate", type=float, default=0.10)
    p.add_argument("--Audit_Alpha0", type=float, default=2.0)
    p.add_argument("--Audit_Beta0", type=float, default=2.0)
    p.add_argument("--Audit_Fail_Penalty", type=float, default=0.08)
    p.add_argument("--Audit_Pass_Recovery", type=float, default=0.03)
    p.add_argument("--Audit_Min_Clean_Streak", type=int, default=3)
    p.add_argument("--Audit_Cooldown_Steps", type=int, default=300)
    p.add_argument("--Audit_Weight_In_Reputation", type=float, default=0.30)
    p.add_argument("--Audit_Low_Truth_Threshold", type=float, default=0.45)
    p.add_argument("--Audit_High_Risk_Threshold", type=float, default=0.65)
    p.add_argument("--Audit_Cooldown_Penalty", type=float, default=0.12)
    p.add_argument("--Audit_Risk_Reward_Penalty", type=float, default=0.30)

    # Oracle and workload settings.
    p.add_argument("--Oracle_Type", type=list, default=[])
    p.add_argument("--Oracle_Cost", type=list, default=[])
    p.add_argument("--Oracle_Acc", type=list, default=[])
    p.add_argument("--Oracle_Tokens", type=list, default=[])
    p.add_argument("--Oracle_Behavior_Probs", type=list, default=[])
    p.add_argument("--Oracle_Validation_Probs", type=list, default=[])
    p.add_argument("--Oracle_Num", type=int, default=15)
    p.add_argument("--Oracles_Per_Type", type=int, default=5)
    p.add_argument("--Service_Type_Num", type=int, default=3)
    p.add_argument("--Oracle_Initial_Reputation", type=float, default=0.5)
    p.add_argument("--Time_Window_Size", type=int, default=5)
    p.add_argument("--Time_Period_Size", type=int, default=60)
    p.add_argument("--Oracle_capacity", type=int, default=1000)

    p.add_argument("--lamda", type=int, default=5)
    p.add_argument("--Request_Num", type=int, default=6000)
    p.add_argument("--Request_len_Mean", type=int, default=6000)
    p.add_argument("--Request_len_Std", type=int, default=500)
    p.add_argument("--Request_ddl", type=float, default=7.0)

    # State, GNN-like encoder, reward, and scenarios.
    p.add_argument("--Noise_Probability", type=float, default=0.0)
    p.add_argument("--Noise_Delay", type=float, default=1.0)
    p.add_argument("--State_Mode", choices=["original", "enhanced"], default="original")
    p.add_argument("--Use_GNN_Encoder", action="store_true")
    p.add_argument("--Disable_GNN_Encoder", action="store_true")
    p.add_argument("--Use_GNN_For_All_RL", action="store_true")
    p.add_argument("--GNN_Message_Steps", type=int, default=2)
    p.add_argument("--GNN_Self_Weight", type=float, default=0.55)
    p.add_argument("--GNN_Neighbor_Weight", type=float, default=0.45)
    p.add_argument("--GNN_Service_Weight", type=float, default=1.00)
    p.add_argument("--GNN_Reliability_Weight", type=float, default=0.45)
    p.add_argument("--GNN_Load_Weight", type=float, default=0.35)
    p.add_argument("--GNN_Cost_Weight", type=float, default=0.25)

    p.add_argument("--Reward_Mode", choices=["original", "risk_aware"], default="original")
    p.add_argument("--Success_Mode", choices=["original", "validation_aware"], default="original")
    p.add_argument("--Scenario", choices=["static", "validation_stress", "rl_hard", "rl_harder"], default="static")
    p.add_argument("--SemiGreedy_View", choices=["myopic", "risk_aware"], default="myopic")
    p.add_argument("--Action_Mask_Mode", choices=["none", "type"], default="none")
    p.add_argument("--Fatigue_Strength", type=float, default=1.0)
    p.add_argument("--Burstiness", type=float, default=0.80)
    p.add_argument("--Expose_Validation_Prob", action="store_true")
    p.add_argument("--Harder_Request_DDL", type=float, default=6.6)

    p.add_argument("--W_SUCCESS", type=float, default=2.2)
    p.add_argument("--W_REPUTATION", type=float, default=0.35)
    p.add_argument("--W_MATCH", type=float, default=0.8)
    p.add_argument("--W_VALIDATION", type=float, default=1.2)
    p.add_argument("--W_COST", type=float, default=1.15)
    p.add_argument("--W_RESPONSE", type=float, default=1.35)
    p.add_argument("--W_BEHAVIOR", type=float, default=0.55)
    p.add_argument("--W_TIMEOUT", type=float, default=2.0)
    p.add_argument("--Reward_Clip", type=float, default=3.0)
    p.add_argument("--Reward_Scale", type=float, default=3.0)

    args = p.parse_args()

    if args.List_Methods:
        print("Available canonical methods:")
        for m in ALL_METHODS:
            print("  -", m)
        print("\nAvailable presets:")
        for k in sorted(METHOD_PRESETS):
            print(f"  - {k}: {' '.join(METHOD_PRESETS[k])}")
        raise SystemExit(0)

    if args.Disable_Audit_Reputation:
        args.Use_Audit_Reputation = False

    if args.Scenario in ["validation_stress", "rl_hard", "rl_harder"]:
        args.Success_Mode = "validation_aware"
        if args.Reward_Mode == "original":
            args.Reward_Mode = "risk_aware"
        if args.State_Mode == "original":
            args.State_Mode = "enhanced"

    if args.Scenario in ["rl_hard", "rl_harder"]:
        if args.Action_Mask_Mode == "none":
            args.Action_Mask_Mode = "type"
        if args.W_SUCCESS == 2.2:
            args.W_SUCCESS = 2.6
        if args.W_RESPONSE == 1.35:
            args.W_RESPONSE = 1.5
        if args.W_TIMEOUT == 2.0:
            args.W_TIMEOUT = 2.2

    if args.Scenario == "rl_harder":
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

    args = _resolve_selected_methods(args)

    if args.Use_HCRL and not args.Disable_GNN_Encoder:
        args.Use_GNN_Encoder = True
    if args.Disable_GNN_Encoder:
        args.Use_GNN_Encoder = False
    if args.Use_GNN_For_All_RL:
        args.Use_GNN_Encoder = True

    if args.COBRA_No_Teacher:
        args.COBRA_Teacher_Source = "none"
        args.COBRA_Teacher_Start_Prob = 0.0
    if args.HCRL_No_Teacher:
        args.HCRL_Teacher_Source = "none"
        args.HCRL_Teacher_Start_Prob = 0.0

    _generate_oracle_community(args)
    args.Baseline_num = len(args.Baselines)

    print(f"[Method selection] mode: {args.Method_Selection_Mode}")
    print(f"[Method selection] running methods: {args.Baselines}")
    print(f"[Audit reputation] enabled: {bool(args.Use_Audit_Reputation)}")
    print(f"[HCRL modes] {args.HCRL_Mode_Names}")
    return args


def _generate_oracle_community(args):
    role_pattern = ["malicious", "trusted_low", "normal", "trusted_mid", "trusted_high"]
    oracle_type = []
    oracle_cost = []
    oracle_acc = []
    oracle_tokens = []
    oracle_behavior_probs = []
    oracle_validation_probs = []
    oracle_fatigue_sensitivity = []
    malicious_index, normal_index, trusted_index = [], [], []

    idx = 0
    for service_type in range(args.Service_Type_Num):
        for k in range(args.Oracles_Per_Type):
            role = role_pattern[k % len(role_pattern)]
            oracle_type.append(service_type)
            if args.Scenario == "validation_stress":
                table = {
                    "malicious":   (0.25, 1.00, 100, [0.40, 0.25, 0.25, 0.10], 0.25, 0.06),
                    "trusted_low": (0.25, 1.00, 150, [0.75, 0.20, 0.05, 0.00], 0.55, 0.08),
                    "normal":      (0.45, 1.10, 250, [0.65, 0.20, 0.15, 0.00], 0.60, 0.05),
                    "trusted_mid": (0.65, 1.15, 400, [0.90, 0.10, 0.00, 0.00], 0.90, 0.02),
                    "trusted_high":(0.95, 1.25, 700, [0.98, 0.02, 0.00, 0.00], 0.98, 0.00),
                }
            elif args.Scenario == "rl_hard":
                table = {
                    "malicious":   (0.18, 1.20,  80, [0.25, 0.25, 0.30, 0.20], 0.28, 0.12),
                    "trusted_low": (0.22, 1.20, 160, [0.70, 0.20, 0.10, 0.00], 0.72, 0.18),
                    "normal":      (0.42, 1.08, 260, [0.62, 0.22, 0.16, 0.00], 0.62, 0.12),
                    "trusted_mid": (0.66, 1.08, 420, [0.88, 0.10, 0.02, 0.00], 0.90, 0.06),
                    "trusted_high":(0.92, 1.12, 720, [0.97, 0.03, 0.00, 0.00], 0.97, 0.02),
                }
            elif args.Scenario == "rl_harder":
                table = {
                    "malicious":   (0.16, 1.28,  80, [0.18, 0.22, 0.35, 0.25], 0.22, 0.20),
                    "trusted_low": (0.20, 1.25, 160, [0.62, 0.24, 0.14, 0.00], 0.68, 0.25),
                    "normal":      (0.40, 1.12, 260, [0.56, 0.24, 0.20, 0.00], 0.58, 0.18),
                    "trusted_mid": (0.66, 1.10, 420, [0.84, 0.13, 0.03, 0.00], 0.88, 0.08),
                    "trusted_high":(0.92, 1.14, 720, [0.96, 0.04, 0.00, 0.00], 0.97, 0.03),
                }
            else:
                table = {
                    "malicious":   (0.20, 1.00, 100, [0.30, 0.25, 0.30, 0.15], 0.30, 0.05),
                    "trusted_low": (0.30, 1.00, 150, [0.75, 0.20, 0.05, 0.00], 0.70, 0.06),
                    "normal":      (0.45, 1.05, 250, [0.65, 0.25, 0.10, 0.00], 0.65, 0.04),
                    "trusted_mid": (0.65, 1.10, 400, [0.90, 0.10, 0.00, 0.00], 0.90, 0.02),
                    "trusted_high":(0.95, 1.20, 700, [0.98, 0.02, 0.00, 0.00], 0.98, 0.00),
                }
            cost, acc, token, probs, val, fatigue = table[role]
            # Tiny deterministic service/type perturbation prevents tied rankings.
            cost = float(cost + 0.005 * (k % 3))
            acc = float(acc + 0.02 * ((service_type + k) % 3))
            oracle_cost.append(cost)
            oracle_acc.append(acc)
            oracle_tokens.append(token)
            oracle_behavior_probs.append(probs)
            oracle_validation_probs.append(val)
            oracle_fatigue_sensitivity.append(fatigue)
            if role == "malicious":
                malicious_index.append(idx)
            elif role == "normal":
                normal_index.append(idx)
            else:
                trusted_index.append(idx)
            idx += 1

    args.Oracle_Type = oracle_type
    args.Oracle_Cost = oracle_cost
    args.Oracle_Acc = oracle_acc
    args.Oracle_Tokens = oracle_tokens
    args.Oracle_Behavior_Probs = oracle_behavior_probs
    args.Oracle_Validation_Probs = oracle_validation_probs
    args.Oracle_Fatigue_Sensitivity = oracle_fatigue_sensitivity
    args.Oracle_Num = len(oracle_type)
    args.Malicious_Oracle_Index = malicious_index
    args.Normal_Oracle_Index = normal_index
    args.Trusted_Oracle_Index = trusted_index
