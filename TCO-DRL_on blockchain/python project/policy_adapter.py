"""
Policy adapter for on-chain deployment.

This module loads offline-trained .npz checkpoints from the simulation code
and exposes one deterministic inference interface for all supported methods.

Important design choice:
    The trained model stays off-chain. Only the selected oracle result is
    submitted to the smart contract.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np


# Ensure we import the enhanced simulation code, not the legacy blockchain copy.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
SIM_DIR = REPO_ROOT / "TCO-DRL_with baseline"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from model import baseline_DQN, DuelingDoubleDQN, OptionActorCritic  # noqa: E402


MODE_SINGLE = 0
MODE_SERIAL = 1
MODE_PARALLEL = 2

METHOD_ID = {
    "DQN": 1,
    "RA-DDQN": 2,
    "PB-SafeDQN": 3,
    "COBRA-Oracle": 4,
    "HCRL-Oracle": 5,
}


@dataclass
class PolicyDecision:
    method: str
    method_id: int
    mode_code: int
    mode_index: int
    mode_name: str
    primary_oracle: int
    backup_oracle: int
    policy_hash: str

    def as_contract_args(self, request_id: int):
        return (
            int(request_id),
            int(self.method_id),
            int(self.mode_code),
            int(self.primary_oracle),
            int(self.backup_oracle),
            bytes.fromhex(self.policy_hash.replace("0x", "")),
        )


def _resolve_path(path_value: str, base_dir: Optional[Path] = None) -> Path:
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = (base_dir or Path.cwd()) / p
    return p.resolve()


def _sha256_files(paths: Dict[str, Path]) -> str:
    h = hashlib.sha256()
    for key in sorted(paths):
        p = paths[key]
        h.update(key.encode("utf-8"))
        h.update(str(p.name).encode("utf-8"))
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return "0x" + h.hexdigest()


def load_npz_checkpoint(model: Any, path: Path, strict: bool = True) -> None:
    """Load arrays saved by model.save_model(...).

    Supported checkpoint parameter names:
      - baseline_DQN / DuelingDoubleDQN: W1, b1, W2, b2 or Wv, bv, Wa, ba
      - OptionActorCritic: W1, b1, Wp, bp, Wv, bv

    The loader intentionally does not restore replay buffers or exploration
    state. On-chain deployment should use deterministic choose_best_action().
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = np.load(str(path), allow_pickle=True)
    loaded = []

    for name in ["W1", "b1", "W2", "b2", "Wv", "bv", "Wa", "ba", "Wp", "bp"]:
        if name in data.files and hasattr(model, name):
            arr = np.asarray(data[name], dtype=np.float32)
            current = getattr(model, name)
            if strict and hasattr(current, "shape") and tuple(current.shape) != tuple(arr.shape):
                raise ValueError(
                    f"Shape mismatch for {path.name}:{name}: "
                    f"checkpoint {arr.shape} != model {current.shape}"
                )
            setattr(model, name, arr)
            loaded.append(name)

    if not loaded:
        raise ValueError(
            f"No compatible parameters found in {path}. "
            f"Available keys: {data.files}"
        )

    if "epsilon" in data.files and hasattr(model, "set_epsilon"):
        # Deterministic inference uses choose_best_action(), but preserving a
        # sane epsilon/temperature is useful for debugging.
        try:
            model.set_epsilon(float(np.asarray(data["epsilon"]).item()))
        except Exception:
            pass

    if hasattr(model, "_copy_eval_to_target"):
        model._copy_eval_to_target()


class OnChainPolicyAdapter:
    """Builds and runs one deployed policy from saved weights."""

    def __init__(self, args, env, method: str, weights: Dict[str, str], config_dir: Optional[Path] = None):
        self.args = args
        self.env = env
        self.method = str(method)
        self.config_dir = config_dir or Path.cwd()

        self.weight_paths: Dict[str, Path] = {
            key: _resolve_path(value, self.config_dir) for key, value in weights.items()
        }
        self.policy_hash = _sha256_files(self.weight_paths)

        self.models: Dict[str, Any] = {}
        self._build_models()
        self._load_weights()

    @property
    def method_id(self) -> int:
        if self.method not in METHOD_ID:
            raise ValueError(f"Unsupported method for on-chain deployment: {self.method}")
        return METHOD_ID[self.method]

    def _build_models(self) -> None:
        a = self.args
        env = self.env

        if self.method == "DQN":
            self.models["primary"] = baseline_DQN(
                env.actionNum,
                env.s_features,
                hidden_units=a.Dqn_hidden,
                scope="DQN",
                learning_rate=a.Dqn_lr,
                memory_size=a.Dqn_memory_size,
                batch_size=a.Dqn_batch_size,
                e_greedy_increment=a.Dqn_epsilon_increment,
                reward_clip=a.Reward_Clip,
                seed=a.Seed,
            )
        elif self.method == "RA-DDQN":
            self.models["primary"] = DuelingDoubleDQN(
                env.actionNum,
                env.s_features,
                hidden_units=a.Dqn_hidden,
                scope="RA_DDQN",
                learning_rate=a.RA_lr,
                memory_size=a.Dqn_memory_size,
                batch_size=a.Dqn_batch_size,
                e_greedy_increment=a.Dqn_epsilon_increment,
                reward_clip=a.Reward_Clip,
                seed=a.Seed + 17,
            )
        elif self.method == "PB-SafeDQN":
            self.models["primary"] = DuelingDoubleDQN(
                env.actionNum,
                env.s_features,
                hidden_units=a.Dqn_hidden,
                scope="PB_SafeDQN",
                learning_rate=a.PB_lr,
                memory_size=a.Dqn_memory_size,
                batch_size=a.Dqn_batch_size,
                e_greedy_increment=a.Dqn_epsilon_increment,
                reward_clip=a.Reward_Clip,
                seed=a.Seed + 1009,
            )
        elif self.method == "COBRA-Oracle":
            self.models["primary"] = DuelingDoubleDQN(
                env.actionNum,
                env.s_features,
                hidden_units=a.Dqn_hidden,
                scope="COBRA_Oracle",
                learning_rate=a.COBRA_lr,
                memory_size=a.Dqn_memory_size,
                batch_size=a.Dqn_batch_size,
                e_greedy_increment=a.Dqn_epsilon_increment,
                reward_clip=a.Reward_Clip,
                seed=a.Seed + 2027,
            )
        elif self.method == "HCRL-Oracle":
            mode_dim = env.s_features + getattr(env, "mode_extra_features", 6)
            n_modes = len(a.HCRL_Mode_Names)

            self.models["mode"] = OptionActorCritic(
                n_modes,
                mode_dim,
                hidden_units=max(64, a.Dqn_hidden // 2),
                scope="HCRL_Mode_OptionAC",
                learning_rate=a.HCRL_Mode_lr,
                memory_size=a.Dqn_memory_size,
                batch_size=a.Dqn_batch_size,
                entropy_coef=a.HCRL_AC_Entropy,
                value_coef=a.HCRL_AC_Value_Coef,
                reward_clip=a.Reward_Clip,
                seed=a.Seed + 3031,
            )
            self.models["primary"] = OptionActorCritic(
                env.actionNum,
                env.s_features,
                hidden_units=a.Dqn_hidden,
                scope="HCRL_Primary_OptionAC",
                learning_rate=a.HCRL_lr,
                memory_size=a.Dqn_memory_size,
                batch_size=a.Dqn_batch_size,
                entropy_coef=a.HCRL_AC_Entropy,
                value_coef=a.HCRL_AC_Value_Coef,
                reward_clip=a.Reward_Clip,
                seed=a.Seed + 4049,
            )
            self.models["backup"] = OptionActorCritic(
                env.actionNum,
                env.s_features,
                hidden_units=a.Dqn_hidden,
                scope="HCRL_Backup_OptionAC",
                learning_rate=a.HCRL_lr,
                memory_size=a.Dqn_memory_size,
                batch_size=a.Dqn_batch_size,
                entropy_coef=a.HCRL_AC_Entropy,
                value_coef=a.HCRL_AC_Value_Coef,
                reward_clip=a.Reward_Clip,
                seed=a.Seed + 5051,
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _load_weights(self) -> None:
        if self.method == "HCRL-Oracle":
            required = ["mode", "primary", "backup"]
        else:
            required = ["primary"]

        missing = [key for key in required if key not in self.weight_paths]
        if missing:
            raise ValueError(
                f"Missing weight path(s) for {self.method}: {missing}. "
                f"Configured weights: {self.weight_paths}"
            )

        for key in required:
            load_npz_checkpoint(self.models[key], self.weight_paths[key], strict=True)

    def _single_oracle_decision(self, request_attrs) -> PolicyDecision:
        state = self.env.getState(request_attrs, self.method)
        mask = self.env.get_action_mask(request_attrs)
        primary = int(self.models["primary"].choose_best_action(state, mask))

        return PolicyDecision(
            method=self.method,
            method_id=self.method_id,
            mode_code=MODE_SINGLE,
            mode_index=0,
            mode_name="single",
            primary_oracle=primary,
            backup_oracle=0,
            policy_hash=self.policy_hash,
        )

    def _primary_backup_decision(self, request_attrs) -> PolicyDecision:
        state = self.env.getState(request_attrs, self.method)
        mask = self.env.get_action_mask(request_attrs)
        primary = int(self.models["primary"].choose_best_action(state, mask))

        backup = -1
        if hasattr(self.env, "choose_backup_oracle"):
            backup = int(self.env.choose_backup_oracle(request_attrs, primary, self.method))
        if backup < 0:
            backup_mask = self.env.get_backup_action_mask(request_attrs, primary)
            valid = np.where(np.asarray(backup_mask, dtype=bool))[0]
            backup = int(valid[0]) if valid.size else primary

        if self.method == "PB-SafeDQN":
            mode_name = getattr(self.args, "PB_Backup_Mode", "parallel")
            mode_code = MODE_SERIAL if mode_name == "serial" else MODE_PARALLEL
        else:
            # COBRA uses adaptive backup gating in the environment. The contract
            # records the selected backup candidate and marks the deployment as
            # serial recovery by default.
            mode_name = "serial_backup"
            mode_code = MODE_SERIAL

        return PolicyDecision(
            method=self.method,
            method_id=self.method_id,
            mode_code=mode_code,
            mode_index=mode_code,
            mode_name=mode_name,
            primary_oracle=primary,
            backup_oracle=backup,
            policy_hash=self.policy_hash,
        )

    def _hcrl_decision(self, request_attrs) -> PolicyDecision:
        primary_state = self.env.getState(request_attrs, "HCRL-Oracle")
        primary_mask = self.env.get_action_mask(request_attrs)
        primary = int(self.models["primary"].choose_best_action(primary_state, primary_mask))

        mode_state = self.env.get_hcrl_mode_state(request_attrs, "HCRL-Oracle")
        if hasattr(self.env, "get_hcrl_mode_mask"):
            mode_mask = self.env.get_hcrl_mode_mask(request_attrs, "HCRL-Oracle", primary)
        else:
            mode_mask = np.ones(len(self.args.HCRL_Mode_Names), dtype=bool)

        mode_index = int(self.models["mode"].choose_best_action(mode_state, mode_mask))
        mode_name = self.args.HCRL_Mode_Names[mode_index]

        if mode_name in ["single_cost", "single_safe"] or mode_name.startswith("single"):
            mode_code = MODE_SINGLE
            backup = 0
        else:
            mode_code = MODE_PARALLEL if mode_name.startswith("parallel") else MODE_SERIAL
            backup_mask = self.env.get_backup_action_mask(request_attrs, primary)
            backup = int(self.models["backup"].choose_best_action(primary_state, backup_mask))

        return PolicyDecision(
            method=self.method,
            method_id=self.method_id,
            mode_code=mode_code,
            mode_index=mode_index,
            mode_name=mode_name,
            primary_oracle=primary,
            backup_oracle=backup,
            policy_hash=self.policy_hash,
        )

    def infer(self, request_attrs) -> PolicyDecision:
        if self.method in ["DQN", "RA-DDQN"]:
            return self._single_oracle_decision(request_attrs)
        if self.method in ["PB-SafeDQN", "COBRA-Oracle"]:
            return self._primary_backup_decision(request_attrs)
        if self.method == "HCRL-Oracle":
            return self._hcrl_decision(request_attrs)
        raise ValueError(f"Unsupported method: {self.method}")

    def update_environment_after_submit(self, request_attrs, decision: PolicyDecision) -> None:
        """Keep the local simulator state consistent after on-chain submission."""
        if self.method in ["DQN", "RA-DDQN"]:
            self.env.feedback(request_attrs, decision.primary_oracle, self.method)
        elif self.method in ["PB-SafeDQN", "COBRA-Oracle"]:
            self.env.feedback_primary_backup(request_attrs, decision.primary_oracle, self.method)
        elif self.method == "HCRL-Oracle":
            self.env.feedback_hcrl(
                request_attrs,
                decision.mode_index,
                decision.primary_oracle,
                decision.backup_oracle,
                self.method,
            )
