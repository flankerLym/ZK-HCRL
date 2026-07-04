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
    "ra": "RA-DDQN", "ra-ddqn": "RA-DDQN", "ra_ddqn": "RA-DDQN", "radqn": "RA-DDQN", "ra-dqn": "RA-DDQN", "ra_dqn": "RA-DDQN",
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


def _ratio_arg(value):
    """Accept 0.3 or 30 as the same malicious ratio."""
    v = float(value)
    if v > 1.0:
        v = v / 100.0
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError("ratio must be in [0, 1] or [0, 100]")
    return v


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

    # Checkpoint/evaluation settings.
    p.add_argument("--Load_Weights", nargs="*", default=[],
                   help="Optional METHOD=path pairs to load trained .npz weights. "
                        "Examples: DQN=output/run/DQN.npz HCRL-Oracle=output/run")
    p.add_argument("--Weight_Path", type=str, default="",
                   help="Shortcut weight path when exactly one selected method is loaded.")
    p.add_argument("--Eval_Only", action="store_true",
                   help="Evaluation-only mode: use greedy actions and do not update replay buffers or train.")
    p.add_argument("--Greedy_Eval", action="store_true",
                   help="Use deterministic best actions for RL policies. Automatically enabled by --Eval_Only.")
    p.add_argument("--Strict_Weight_Load", action="store_true",
                   help="Raise an error if a checkpoint does not match the instantiated model shape.")

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

    # New: parameterized malicious-oracle ratio.
    p.add_argument("--Malicious_Ratio", "--Malicious_Rate", "--Malicious_Percent",
                   type=_ratio_arg, default=None,
                   help="Override malicious oracle ratio. Accepts 0.3 or 30 for 30%%. If omitted, keeps the original role_pattern.")
    p.add_argument("--Malicious_Placement", choices=["balanced", "front", "random"], default="balanced",
                   help="How to place malicious oracle indices when --Malicious_Ratio is set. balanced keeps service types comparable.")
    p.add_argument("--Malicious_Seed", type=int, default=None,
                   help="Seed used only for --Malicious_Placement random. Defaults to --Seed.")

    # Attack-scenario stress-test settings. These modify only malicious-oracle
    # behavior after the normal oracle community is generated.
    p.add_argument("--Attack_Profile", choices=["none", "mild", "stealth", "mixed", "severe", "burst"],
                   default="none",
                   help="Adversarial profile for malicious oracles during attack-scenario experiments.")
    p.add_argument("--Attack_Intensity", type=float, default=1.0,
                   help="Multiplier for attack severity; 1.0 keeps the predefined profile.")

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
    if args.Eval_Only:
        args.Greedy_Eval = True

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
    _apply_attack_profile(args)
    args.Baseline_num = len(args.Baselines)

    print(f"[Method selection] mode: {args.Method_Selection_Mode}")
    print(f"[Method selection] running methods: {args.Baselines}")
    print(f"[Audit reputation] enabled: {bool(args.Use_Audit_Reputation)}")
    print(f"[HCRL modes] {args.HCRL_Mode_Names}")
    print(f"[Oracle community] total={args.Oracle_Num}, service_types={args.Service_Type_Num}, "
          f"oracles_per_type={args.Oracles_Per_Type}, malicious={len(args.Malicious_Oracle_Index)} "
          f"({args.Malicious_Ratio_Effective:.3f}), normal={len(args.Normal_Oracle_Index)}, "
          f"trusted={len(args.Trusted_Oracle_Index)}, placement={args.Malicious_Placement if args.Malicious_Ratio is not None else 'original_role_pattern'}")
    print(f"[Attack profile] profile={args.Attack_Profile}, intensity={args.Attack_Intensity}")
    print(f"[Eval/checkpoint] eval_only={bool(args.Eval_Only)}, greedy_eval={bool(args.Greedy_Eval)}, load_weights={args.Load_Weights or args.Weight_Path}")
    return args


def _scenario_role_table(scenario):
    if scenario == "validation_stress":
        return {
            "malicious":   (0.25, 1.00, 100, [0.40, 0.25, 0.25, 0.10], 0.25, 0.06),
            "trusted_low": (0.25, 1.00, 150, [0.75, 0.20, 0.05, 0.00], 0.55, 0.08),
            "normal":      (0.45, 1.10, 250, [0.65, 0.20, 0.15, 0.00], 0.60, 0.05),
            "trusted_mid": (0.65, 1.15, 400, [0.90, 0.10, 0.00, 0.00], 0.90, 0.02),
            "trusted_high":(0.95, 1.25, 700, [0.98, 0.02, 0.00, 0.00], 0.98, 0.00),
        }
    if scenario == "rl_hard":
        return {
            "malicious":   (0.18, 1.20,  80, [0.25, 0.25, 0.30, 0.20], 0.28, 0.12),
            "trusted_low": (0.22, 1.20, 160, [0.70, 0.20, 0.10, 0.00], 0.72, 0.18),
            "normal":      (0.42, 1.08, 260, [0.62, 0.22, 0.16, 0.00], 0.62, 0.12),
            "trusted_mid": (0.66, 1.08, 420, [0.88, 0.10, 0.02, 0.00], 0.90, 0.06),
            "trusted_high":(0.92, 1.12, 720, [0.97, 0.03, 0.00, 0.00], 0.97, 0.02),
        }
    if scenario == "rl_harder":
        return {
            "malicious":   (0.16, 1.28,  80, [0.18, 0.22, 0.35, 0.25], 0.22, 0.20),
            "trusted_low": (0.20, 1.25, 160, [0.62, 0.24, 0.14, 0.00], 0.68, 0.25),
            "normal":      (0.40, 1.12, 260, [0.56, 0.24, 0.20, 0.00], 0.58, 0.18),
            "trusted_mid": (0.66, 1.10, 420, [0.84, 0.13, 0.03, 0.00], 0.88, 0.08),
            "trusted_high":(0.92, 1.14, 720, [0.96, 0.04, 0.00, 0.00], 0.97, 0.03),
        }
    return {
        "malicious":   (0.20, 1.00, 100, [0.30, 0.25, 0.30, 0.15], 0.30, 0.05),
        "trusted_low": (0.30, 1.00, 150, [0.75, 0.20, 0.05, 0.00], 0.70, 0.06),
        "normal":      (0.45, 1.05, 250, [0.65, 0.25, 0.10, 0.00], 0.65, 0.04),
        "trusted_mid": (0.65, 1.10, 400, [0.90, 0.10, 0.00, 0.00], 0.90, 0.02),
        "trusted_high":(0.95, 1.20, 700, [0.98, 0.02, 0.00, 0.00], 0.98, 0.00),
    }


def _build_malicious_sets(args, total_oracles):
    """Return malicious global-index set and optional per-service local malicious sets.

    If --Malicious_Ratio is omitted, return None to preserve the original fixed
    role pattern: [malicious, trusted_low, normal, trusted_mid, trusted_high].
    """
    if args.Malicious_Ratio is None:
        return None, None

    target = int(np.floor(total_oracles * float(args.Malicious_Ratio) + 0.5))
    target = int(np.clip(target, 0, total_oracles))

    if args.Malicious_Placement == "front":
        return set(range(target)), None

    if args.Malicious_Placement == "random":
        rng = np.random.RandomState(args.Seed if args.Malicious_Seed is None else args.Malicious_Seed)
        if target == 0:
            return set(), None
        return set(map(int, rng.choice(np.arange(total_oracles), size=target, replace=False))), None

    # balanced placement: distribute malicious nodes across service types first,
    # so each service type has a comparable malicious ratio.
    per_service = {s: set() for s in range(args.Service_Type_Num)}
    remaining = target
    local = 0
    while remaining > 0:
        any_added = False
        for s in range(args.Service_Type_Num):
            if remaining <= 0:
                break
            if local < args.Oracles_Per_Type and len(per_service[s]) < args.Oracles_Per_Type:
                per_service[s].add(local)
                remaining -= 1
                any_added = True
        local += 1
        if not any_added:
            break
    global_set = set()
    for s, locals_ in per_service.items():
        base_idx = s * args.Oracles_Per_Type
        global_set.update(base_idx + k for k in locals_)
    return global_set, per_service


def _generate_oracle_community(args):
    original_role_pattern = ["malicious", "trusted_low", "normal", "trusted_mid", "trusted_high"]
    non_malicious_pattern = ["trusted_low", "normal", "trusted_mid", "trusted_high"]
    table = _scenario_role_table(args.Scenario)

    oracle_type = []
    oracle_cost = []
    oracle_acc = []
    oracle_tokens = []
    oracle_behavior_probs = []
    oracle_validation_probs = []
    oracle_fatigue_sensitivity = []
    malicious_index, normal_index, trusted_index = [], [], []

    total_oracles = int(args.Service_Type_Num) * int(args.Oracles_Per_Type)
    malicious_set, per_service_malicious = _build_malicious_sets(args, total_oracles)
    non_mal_counter_by_service = {s: 0 for s in range(args.Service_Type_Num)}

    idx = 0
    for service_type in range(args.Service_Type_Num):
        for k in range(args.Oracles_Per_Type):
            if malicious_set is None:
                role = original_role_pattern[k % len(original_role_pattern)]
            else:
                if idx in malicious_set:
                    role = "malicious"
                else:
                    c = non_mal_counter_by_service[service_type]
                    role = non_malicious_pattern[c % len(non_malicious_pattern)]
                    non_mal_counter_by_service[service_type] += 1

            oracle_type.append(service_type)
            cost, acc, token, probs, val, fatigue = table[role]
            # Tiny deterministic service/type perturbation prevents tied rankings.
            cost = float(cost + 0.005 * (k % 3))
            acc = float(acc + 0.02 * ((service_type + k) % 3))
            oracle_cost.append(cost)
            oracle_acc.append(acc)
            oracle_tokens.append(token)
            oracle_behavior_probs.append(list(probs))
            oracle_validation_probs.append(float(val))
            oracle_fatigue_sensitivity.append(float(fatigue))
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
    args.Malicious_Ratio_Effective = len(malicious_index) / max(float(args.Oracle_Num), 1.0)


def _apply_attack_profile(args):
    """Modify malicious-oracle behavior for attack-scenario experiments.

    The default oracle community already defines malicious behavior for each
    scenario. Attack_Profile provides controlled post-generation stress tests
    while keeping the service types and oracle count unchanged.
    """
    profile = getattr(args, "Attack_Profile", "none")
    if profile in (None, "none"):
        return
    malicious = list(getattr(args, "Malicious_Oracle_Index", []))
    if not malicious:
        return
    intensity = float(np.clip(getattr(args, "Attack_Intensity", 1.0), 0.0, 3.0))

    # behavior probs are [normal, small abnormal, medium abnormal, severe abnormal]
    profiles = {
        "mild":   ([0.35, 0.30, 0.25, 0.10], 0.32, 0.15),
        "stealth":([0.62, 0.20, 0.13, 0.05], 0.42, 0.08),
        "mixed":  ([0.25, 0.25, 0.30, 0.20], 0.25, 0.20),
        "severe": ([0.08, 0.17, 0.35, 0.40], 0.12, 0.35),
        "burst":  ([0.18, 0.22, 0.30, 0.30], 0.20, 0.55),
    }
    base_probs, base_validation, base_fatigue = profiles[profile]

    # Intensity shifts probability mass from normal behavior to severe behavior.
    probs = np.asarray(base_probs, dtype=float)
    if intensity != 1.0:
        shift = min(max(intensity - 1.0, -0.8), 2.0)
        if shift >= 0:
            move = min(probs[0] * 0.65, 0.12 * shift)
            probs[0] -= move
            probs[-1] += move
        else:
            move = min(probs[-1] * 0.60, 0.10 * abs(shift))
            probs[-1] -= move
            probs[0] += move
    probs = np.clip(probs, 1e-6, None)
    probs = (probs / probs.sum()).tolist()
    validation = float(np.clip(base_validation / max(intensity, 0.2), 0.02, 0.95))
    fatigue = float(np.clip(base_fatigue * max(intensity, 0.1), 0.0, 1.5))

    for idx in malicious:
        args.Oracle_Behavior_Probs[idx] = list(probs)
        args.Oracle_Validation_Probs[idx] = validation
        if hasattr(args, "Oracle_Fatigue_Sensitivity"):
            args.Oracle_Fatigue_Sensitivity[idx] = fatigue

    if profile == "burst":
        args.Burstiness = max(float(args.Burstiness), min(0.98, 0.90 + 0.02 * intensity))
        args.Fatigue_Strength = max(float(args.Fatigue_Strength), 2.2 + 0.6 * intensity)
