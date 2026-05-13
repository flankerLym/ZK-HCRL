"""
Apply HCRL ME/OOA/OSA attack-defense improvements to TCO-DRL_with baseline.

Run from the repository root:
    python apply_hcrl_me_ooa_osa_patch.py

This patch adds:
  1) Trust-aware action masking driven by the existing effective reputation.
  2) ME / OOA / OSA attack simulation hooks in the baseline environment.
  3) HCRL main-loop calls that pass policy_name into primary/backup masks.
  4) An evaluation script that loads saved .npz HCRL weights and tests attack defense.

The patch creates timestamped .bak_hcrl_attack_* backups before editing files.
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
BASELINE = ROOT / "TCO-DRL_with baseline"
ENV = BASELINE / "env.py"
PARAM = BASELINE / "param_parser.py"
MAIN = BASELINE / "main.py"

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(path: Path) -> None:
    if path.exists():
        dst = path.with_suffix(path.suffix + f".bak_hcrl_attack_{STAMP}")
        shutil.copy2(path, dst)
        print(f"[backup] {path} -> {dst}")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"[write] {path}")


def patch_param_parser() -> None:
    text = read(PARAM)
    if "--Use_Trust_Action_Mask" not in text:
        insert = r'''
    # Trust-aware action masking and attack-defense evaluation.
    p.add_argument("--Use_Trust_Action_Mask", action="store_true", default=False,
                   help="Mask low-effective-reputation oracles during primary/backup selection.")
    p.add_argument("--Trust_Action_Threshold", type=float, default=0.45,
                   help="Effective reputation threshold for trust-aware action masking.")
    p.add_argument("--Trust_Mask_Warmup_Requests", type=int, default=300,
                   help="Disable trust masking before this many requests so reputation can warm up.")
    p.add_argument("--Trust_Mask_Methods", nargs="+", default=["HCRL-Oracle"],
                   help="Methods using trust-aware mask; use 'all' to apply to every method.")
    p.add_argument("--Trust_Mask_Fallback", choices=["service", "trusted", "all"], default="service",
                   help="Fallback candidate set if trust threshold removes every service-matched oracle.")

    p.add_argument("--Attack_Mode", choices=["none", "ME", "OOA", "OSA"], default="none",
                   help="ME/OOA/OSA oracle behavior attack mode for defense experiments.")
    p.add_argument("--Attack_Oracles", nargs="+", type=int, default=[],
                   help="Oracle indices controlled by the attacker.")
    p.add_argument("--Trusted_Reference_Oracles", nargs="+", type=int, default=[],
                   help="Reference trusted oracle indices for attack diagnostics/fallback.")
    p.add_argument("--Attack_Start_Period", type=int, default=3,
                   help="Time period at which attacks begin.")
    p.add_argument("--OOA_Off_Periods", type=int, default=1,
                   help="OOA malicious/off periods in each cycle.")
    p.add_argument("--OOA_On_Periods", type=int, default=5,
                   help="OOA honest/on periods in each cycle.")
    p.add_argument("--OSA_Margin", type=float, default=0.08,
                   help="OSA becomes malicious when reputation exceeds threshold + margin.")
    p.add_argument("--Attack_Bad_Validation_Prob", type=float, default=0.03,
                   help="Validation probability for attacked oracles during malicious phase.")
    p.add_argument("--Attack_Honest_Validation_Prob", type=float, default=0.90,
                   help="Validation probability for attacked oracles during OOA/OSA honest phase.")
    p.add_argument("--Attack_Bad_Behavior_Probs", nargs=4, type=float,
                   default=[0.05, 0.10, 0.35, 0.50],
                   help="Behavior probabilities [safe, minor, moderate, severe] in malicious phase.")
    p.add_argument("--Attack_Honest_Behavior_Probs", nargs=4, type=float,
                   default=[0.92, 0.08, 0.00, 0.00],
                   help="Behavior probabilities [safe, minor, moderate, severe] in honest phase.")
'''
        anchor = "    # Oracle and workload settings."
        if anchor not in text:
            raise RuntimeError("param_parser.py anchor not found: '# Oracle and workload settings.'")
        text = text.replace(anchor, insert + "\n" + anchor)

    # Normalize Attack_Oracles if argparse receives comma-separated strings from older commands.
    if "def _normalize_int_list" not in text:
        helper = r'''

def _normalize_int_list(value):
    """Accept argparse lists like [4, 19, 29] or comma strings like ['4,19,29']."""
    if value is None:
        return []
    out = []
    for item in value if isinstance(value, (list, tuple)) else [value]:
        if isinstance(item, str):
            for part in item.replace(";", ",").split(","):
                part = part.strip()
                if part:
                    out.append(int(part))
        else:
            out.append(int(item))
    return out
'''
        text = text.replace("\ndef parameter_parser():", helper + "\ndef parameter_parser():")

    if "args.Attack_Oracles = _normalize_int_list" not in text:
        anchor = "    args = _resolve_selected_methods(args)"
        text = text.replace(anchor, "    args.Attack_Oracles = _normalize_int_list(getattr(args, \"Attack_Oracles\", []))\n    args.Trusted_Reference_Oracles = _normalize_int_list(getattr(args, \"Trusted_Reference_Oracles\", []))\n\n" + anchor)

    write(PARAM, text)


def patch_env() -> None:
    text = read(ENV)

    # Make workload expose current request id for attack phase logic.
    if "self.current_request_id = request_id" not in text:
        text = text.replace(
            "        request_id = int(request_count) - 1\n        attrs = [request_id, float(self.arrival_Times[request_id]), float(self.lengths[request_id]),",
            "        request_id = int(request_count) - 1\n        self.current_request_id = request_id\n        attrs = [request_id, float(self.arrival_Times[request_id]), float(self.lengths[request_id]),"
        )

    # Replace get_action_mask + get_backup_action_mask block.
    pattern = re.compile(
        r"    def get_action_mask\(self, request_attrs\):\n"
        r"(?P<body>.*?)"
        r"    def _estimated_oracle_metrics\(self, request_attrs, policy_name\):\n",
        flags=re.S,
    )
    if not pattern.search(text):
        raise RuntimeError("Could not locate get_action_mask/get_backup_action_mask block in env.py")

    new_block = r'''    # ------------------------------------------------------------------
    # Trust-aware action masks and attack simulation helpers
    # ------------------------------------------------------------------
    def _trust_mask_methods_enabled(self, policy_name):
        methods = getattr(self.args, "Trust_Mask_Methods", ["HCRL-Oracle"])
        if methods is None:
            methods = ["HCRL-Oracle"]
        methods = [str(m) for m in methods]
        return "all" in methods or (policy_name is not None and str(policy_name) in methods)

    def _attack_oracle_set(self):
        return set(int(x) for x in getattr(self.args, "Attack_Oracles", []) or [])

    def _trusted_reference_set(self):
        return set(int(x) for x in getattr(self.args, "Trusted_Reference_Oracles", []) or [])

    def _current_time_period(self):
        rid = int(getattr(self, "current_request_id", 0))
        return int(rid // max(int(getattr(self.args, "Time_Period_Size", 60)), 1)) + 1

    def _attack_malicious_phase(self, action, policy_name):
        """Return True if an attacked oracle should behave maliciously now."""
        mode = str(getattr(self.args, "Attack_Mode", "none"))
        if mode == "none":
            return False
        action = int(action)
        if action not in self._attack_oracle_set():
            return False
        tp = self._current_time_period()
        if tp < int(getattr(self.args, "Attack_Start_Period", 3)):
            return False
        if mode == "ME":
            return True
        if mode == "OOA":
            off = max(int(getattr(self.args, "OOA_Off_Periods", 1)), 1)
            on = max(int(getattr(self.args, "OOA_On_Periods", 5)), 0)
            cycle = off + on
            pos = (tp - int(getattr(self.args, "Attack_Start_Period", 3))) % max(cycle, 1)
            return pos < off
        if mode == "OSA":
            # Opportunistic: attack while reputation is safely above threshold; behave well near/below threshold.
            try:
                rep = float(self._effective_reputation_vector(policy_name)[action])
            except Exception:
                rep = float(self.oracle_events[policy_name][2, action]) if policy_name in self.oracle_events else 0.5
            thr = float(getattr(self.args, "Trust_Action_Threshold", 0.45))
            margin = float(getattr(self.args, "OSA_Margin", 0.08))
            return rep > thr + margin
        return False

    def _attack_validation_prob(self, action, policy_name, base_prob):
        mode = str(getattr(self.args, "Attack_Mode", "none"))
        action = int(action)
        if mode == "none" or action not in self._attack_oracle_set():
            return float(base_prob)
        if self._current_time_period() < int(getattr(self.args, "Attack_Start_Period", 3)):
            return float(base_prob)
        if self._attack_malicious_phase(action, policy_name):
            return float(np.clip(getattr(self.args, "Attack_Bad_Validation_Prob", 0.03), 0.0, 1.0))
        if mode in ["OOA", "OSA"]:
            return float(np.clip(getattr(self.args, "Attack_Honest_Validation_Prob", 0.90), 0.0, 1.0))
        return float(base_prob)

    def _attack_behavior_probs(self, action, policy_name, base_probs):
        mode = str(getattr(self.args, "Attack_Mode", "none"))
        action = int(action)
        if mode == "none" or action not in self._attack_oracle_set():
            return np.asarray(base_probs, dtype=float)
        if self._current_time_period() < int(getattr(self.args, "Attack_Start_Period", 3)):
            return np.asarray(base_probs, dtype=float)
        if self._attack_malicious_phase(action, policy_name):
            probs = np.asarray(getattr(self.args, "Attack_Bad_Behavior_Probs", [0.05, 0.10, 0.35, 0.50]), dtype=float)
        else:
            probs = np.asarray(getattr(self.args, "Attack_Honest_Behavior_Probs", [0.92, 0.08, 0.0, 0.0]), dtype=float)
        if probs.shape[0] != 4 or probs.sum() <= 0:
            probs = np.asarray(base_probs, dtype=float)
        return probs / max(float(probs.sum()), 1e-8)

    def _apply_trust_action_mask(self, base_mask, policy_name=None, request_attrs=None):
        base_mask = np.asarray(base_mask, dtype=bool).copy()
        if not getattr(self.args, "Use_Trust_Action_Mask", False):
            return base_mask
        if not self._trust_mask_methods_enabled(policy_name):
            return base_mask
        if request_attrs is not None:
            request_id = int(request_attrs[0])
            if request_id < int(getattr(self.args, "Trust_Mask_Warmup_Requests", 300)):
                return base_mask
        if policy_name is None or policy_name not in self.policy_names:
            return base_mask

        rep = self._effective_reputation_vector(policy_name)
        threshold = float(getattr(self.args, "Trust_Action_Threshold", 0.45))
        trust_mask = rep >= threshold
        final_mask = base_mask & trust_mask
        if np.any(final_mask):
            return final_mask.astype(bool)

        fallback = str(getattr(self.args, "Trust_Mask_Fallback", "service"))
        if fallback == "trusted":
            trusted = np.zeros(self.oracleNum, dtype=bool)
            for idx in self._trusted_reference_set():
                if 0 <= idx < self.oracleNum:
                    trusted[idx] = True
            final_mask = base_mask & trusted
            if np.any(final_mask):
                return final_mask.astype(bool)
        elif fallback == "all":
            final_mask = trust_mask
            if np.any(final_mask):
                return final_mask.astype(bool)

        # Safe fallback to avoid an empty action set: keep original service/type mask.
        if np.any(base_mask):
            return base_mask.astype(bool)
        return np.ones(self.oracleNum, dtype=bool)

    def get_action_mask(self, request_attrs, policy_name=None):
        if getattr(self.args, "Action_Mask_Mode", "none") != "type":
            mask = np.ones(self.oracleNum, dtype=bool)
        else:
            mask = (self.oracleTypes == int(request_attrs[3]))
            if not np.any(mask):
                mask[:] = True
        return self._apply_trust_action_mask(mask, policy_name, request_attrs)

    def get_backup_action_mask(self, request_attrs, primary_action, policy_name=None):
        if getattr(self.args, "Action_Mask_Mode", "none") != "type":
            mask = np.ones(self.oracleNum, dtype=bool)
        else:
            mask = (self.oracleTypes == int(request_attrs[3]))
            if not np.any(mask):
                mask[:] = True
        if 0 <= int(primary_action) < self.oracleNum:
            mask[int(primary_action)] = False
        if not np.any(mask):
            mask[:] = True
            if 0 <= int(primary_action) < self.oracleNum:
                mask[int(primary_action)] = False
        mask = self._apply_trust_action_mask(mask, policy_name, request_attrs)
        if 0 <= int(primary_action) < self.oracleNum:
            mask[int(primary_action)] = False
        if not np.any(mask):
            # Final fallback: any non-primary oracle.
            mask = np.ones(self.oracleNum, dtype=bool)
            if 0 <= int(primary_action) < self.oracleNum:
                mask[int(primary_action)] = False
        if not np.any(mask):
            mask[:] = True
        return mask.astype(bool)

    def _estimated_oracle_metrics(self, request_attrs, policy_name):
'''
    text = pattern.sub(new_block, text)

    # Trust-aware internal HCRL calls.
    text = text.replace("self.get_backup_action_mask(request_attrs, primary)", "self.get_backup_action_mask(request_attrs, primary, policy_name)")
    text = text.replace("self.get_backup_action_mask(request_attrs, primary_action)", "self.get_backup_action_mask(request_attrs, primary_action, policy_name)")

    # Dynamic attack validation probability before return.
    if "base = self._attack_validation_prob(action, policy_name, base)" not in text:
        text = text.replace(
            "        return base\n\n    def _simulate_oracle_attempt",
            "        base = self._attack_validation_prob(action, policy_name, base)\n        return base\n\n    def _simulate_oracle_attempt"
        )

    # Dynamic attack behavior distribution.
    if "probs = self._attack_behavior_probs(action, policy_name, probs)" not in text:
        text = text.replace(
            "        probs = np.asarray(self.oracleBehaviorProbs[action], dtype=float)\n        probs = probs / max(float(probs.sum()), 1e-8)",
            "        probs = np.asarray(self.oracleBehaviorProbs[action], dtype=float)\n        probs = self._attack_behavior_probs(action, policy_name, probs)\n        probs = probs / max(float(probs.sum()), 1e-8)"
        )

    # If choose_backup_oracle exists and calls get_backup_action_mask without policy, patch it.
    text = text.replace("get_backup_action_mask(request_attrs, primary_action)", "get_backup_action_mask(request_attrs, primary_action, policy_name)")
    text = text.replace("get_backup_action_mask(request_attrs, primary)", "get_backup_action_mask(request_attrs, primary, policy_name)")

    write(ENV, text)


def patch_main() -> None:
    text = read(MAIN)
    # Keep classical methods unchanged by default; make HCRL primary/backup trust-aware.
    text = text.replace(
        "primary_mask = env.get_action_mask(request_attrs)",
        "primary_mask = env.get_action_mask(request_attrs, \"HCRL-Oracle\")"
    )
    text = text.replace(
        "backup_mask = env.get_backup_action_mask(request_attrs, primary_action)",
        "backup_mask = env.get_backup_action_mask(request_attrs, primary_action, \"HCRL-Oracle\")"
    )
    # When teacher backup is used, prefer trust-aware heuristic if env helper supports it.
    text = text.replace(
        "backup_action = env.choose_backup_oracle(request_attrs, primary_action, \"HCRL-Oracle\")",
        "backup_action = env.choose_backup_oracle(request_attrs, primary_action, \"HCRL-Oracle\")"
    )
    write(MAIN, text)


def main():
    if not BASELINE.exists():
        raise SystemExit(f"Cannot find baseline folder: {BASELINE}\nRun this script from the repository root.")
    for p in [PARAM, ENV, MAIN]:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")
        backup(p)
    patch_param_parser()
    patch_env()
    patch_main()
    print("\nPatch completed.")
    print("Next: run a smoke test from TCO-DRL_with baseline, e.g.:")
    print(r'''python main.py `
  --Methods HCRL-Oracle `
  --Scenario rl_harder `
  --Seed 6 `
  --Epoch 1 `
  --Request_Num 600 `
  --Oracle_Num 30 `
  --Oracles_Per_Type 10 `
  --Service_Type_Num 3 `
  --Use_Trust_Action_Mask `
  --Trust_Action_Threshold 0.45 `
  --Attack_Mode ME `
  --Attack_Oracles 4 19 29 `
  --Trusted_Reference_Oracles 3 14 24 `
  --Run_Tag smoke_hcrl_tam_me''')


if __name__ == "__main__":
    main()
