import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import random
from collections import deque
import numpy as np


class baseline_DQN:
    """Crash-safe NumPy DQN with optional Double/Dueling heads.

    Public API matches the original project:
      choose_action, choose_best_action, store_transition, learn, copy_from,
      set_epsilon, save_model.
    """
    def __init__(self, n_actions, n_features, learning_rate=0.003, reward_decay=0.9,
                 e_greedy=0.9, replace_target_iter=200, memory_size=800, batch_size=30,
                 e_greedy_increment=0.001, hidden_units=64, scope="DQN",
                 double_dqn=False, dueling=False, reward_clip=20.0, grad_clip=5.0, seed=6):
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.lr = float(learning_rate)
        self.gamma = float(reward_decay)
        self.epsilon_max = float(e_greedy)
        self.replace_target_iter = int(replace_target_iter)
        self.memory_size = int(memory_size)
        self.batch_size = int(batch_size)
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max
        self.hidden_units = int(hidden_units)
        self.scope = str(scope)
        self.double_dqn = bool(double_dqn)
        self.dueling = bool(dueling)
        self.reward_clip = float(reward_clip)
        self.grad_clip = float(grad_clip)
        offset = sum(ord(c) for c in self.scope) % 10000
        self.rng = np.random.RandomState(seed + offset)
        self.learn_step_counter = 0
        self.replay_buffer = deque(maxlen=self.memory_size)
        self.reward_list = []
        self._init_params()
        self._copy_eval_to_target()

    def _init_params(self):
        lim1 = np.sqrt(6.0 / (self.n_features + self.hidden_units))
        self.W1 = self.rng.uniform(-lim1, lim1, (self.n_features, self.hidden_units)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_units, dtype=np.float32)
        if self.dueling:
            self.Wv = self.rng.uniform(-0.1, 0.1, (self.hidden_units, 1)).astype(np.float32)
            self.bv = np.zeros(1, dtype=np.float32)
            self.Wa = self.rng.uniform(-0.1, 0.1, (self.hidden_units, self.n_actions)).astype(np.float32)
            self.ba = np.zeros(self.n_actions, dtype=np.float32)
        else:
            self.W2 = self.rng.uniform(-0.1, 0.1, (self.hidden_units, self.n_actions)).astype(np.float32)
            self.b2 = np.zeros(self.n_actions, dtype=np.float32)

    def _copy_eval_to_target(self):
        self.tW1 = self.W1.copy(); self.tb1 = self.b1.copy()
        if self.dueling:
            self.tWv = self.Wv.copy(); self.tbv = self.bv.copy()
            self.tWa = self.Wa.copy(); self.tba = self.ba.copy()
        else:
            self.tW2 = self.W2.copy(); self.tb2 = self.b2.copy()

    def _sanitize_state(self, state):
        x = np.asarray(state, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        scale = max(float(np.max(np.abs(x))), 1.0)
        return np.clip(x / scale, -10.0, 10.0).astype(np.float32)

    def _sanitize_mask(self, action_mask=None):
        if action_mask is None:
            return np.ones(self.n_actions, dtype=bool)
        mask = np.asarray(action_mask, dtype=bool)
        if mask.shape[0] != self.n_actions or not np.any(mask):
            return np.ones(self.n_actions, dtype=bool)
        return mask

    def _forward(self, X, target=False, return_cache=False):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        W1, b1 = (self.tW1, self.tb1) if target else (self.W1, self.b1)
        Z1 = X.dot(W1) + b1
        H = np.maximum(Z1, 0.0)
        if self.dueling:
            if target:
                V = H.dot(self.tWv) + self.tbv
                A = H.dot(self.tWa) + self.tba
            else:
                V = H.dot(self.Wv) + self.bv
                A = H.dot(self.Wa) + self.ba
            Q = V + A - np.mean(A, axis=1, keepdims=True)
        else:
            W2, b2 = (self.tW2, self.tb2) if target else (self.W2, self.b2)
            Q = H.dot(W2) + b2
        Q = np.nan_to_num(Q, nan=0.0, posinf=1e3, neginf=-1e3)
        if return_cache:
            return Q, (X, Z1, H)
        return Q

    def choose_action(self, observation, action_mask=None):
        x = self._sanitize_state(observation)
        valid = np.where(self._sanitize_mask(action_mask))[0]
        if self.rng.uniform() < self.epsilon:
            q = self._forward(x)[0]
            mq = np.full(self.n_actions, -1e9, dtype=np.float32)
            mq[valid] = q[valid]
            return int(np.argmax(mq))
        return int(self.rng.choice(valid))

    def choose_best_action(self, observation, action_mask=None):
        x = self._sanitize_state(observation)
        q = self._forward(x)[0]
        mask = self._sanitize_mask(action_mask)
        mq = np.full(self.n_actions, -1e9, dtype=np.float32)
        mq[mask] = q[mask]
        return int(np.argmax(mq))

    def store_transition(self, s, a, r, s_, next_action_mask=None):
        self.replay_buffer.append((
            self._sanitize_state(s), int(a), float(np.clip(r, -self.reward_clip, self.reward_clip)),
            self._sanitize_state(s_), self._sanitize_mask(next_action_mask),
        ))

    def _clip(self, g):
        g = np.nan_to_num(g, nan=0.0, posinf=self.grad_clip, neginf=-self.grad_clip)
        n = np.linalg.norm(g)
        return g * (self.grad_clip / (n + 1e-8)) if n > self.grad_clip else g

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._copy_eval_to_target()
        batch = random.sample(self.replay_buffer, self.batch_size)
        S = np.asarray([b[0] for b in batch], dtype=np.float32)
        A = np.asarray([b[1] for b in batch], dtype=np.int32)
        R = np.asarray([b[2] for b in batch], dtype=np.float32)
        S2 = np.asarray([b[3] for b in batch], dtype=np.float32)
        masks2 = np.asarray([b[4] for b in batch], dtype=bool)
        q_next_t = self._forward(S2, target=True)
        if self.double_dqn:
            q_next_eval = self._forward(S2, target=False)
            best_next = np.argmax(np.where(masks2, q_next_eval, -1e9), axis=1)
            y = R + self.gamma * q_next_t[np.arange(self.batch_size), best_next]
        else:
            y = R + self.gamma * np.max(np.where(masks2, q_next_t, -1e9), axis=1)
        y = np.clip(np.nan_to_num(y), -self.reward_clip, self.reward_clip)

        q, cache = self._forward(S, target=False, return_cache=True)
        X, Z1, H = cache
        err = np.clip((q[np.arange(self.batch_size), A] - y) / self.batch_size, -5, 5).astype(np.float32)
        dQ = np.zeros_like(q, dtype=np.float32)
        dQ[np.arange(self.batch_size), A] = err
        if self.dueling:
            dV = np.sum(dQ, axis=1, keepdims=True)
            dA = dQ - np.mean(dQ, axis=1, keepdims=True)
            gWv = H.T.dot(dV); gbv = np.sum(dV, axis=0)
            gWa = H.T.dot(dA); gba = np.sum(dA, axis=0)
            dH = dV.dot(self.Wv.T) + dA.dot(self.Wa.T)
        else:
            gW2 = H.T.dot(dQ); gb2 = np.sum(dQ, axis=0)
            dH = dQ.dot(self.W2.T)
        dZ1 = dH * (Z1 > 0)
        gW1 = X.T.dot(dZ1); gb1 = np.sum(dZ1, axis=0)
        self.W1 -= self.lr * self._clip(gW1).astype(np.float32)
        self.b1 -= self.lr * self._clip(gb1).astype(np.float32)
        if self.dueling:
            self.Wv -= self.lr * self._clip(gWv).astype(np.float32); self.bv -= self.lr * self._clip(gbv).astype(np.float32)
            self.Wa -= self.lr * self._clip(gWa).astype(np.float32); self.ba -= self.lr * self._clip(gba).astype(np.float32)
        else:
            self.W2 -= self.lr * self._clip(gW2).astype(np.float32); self.b2 -= self.lr * self._clip(gb2).astype(np.float32)
        for name in vars(self):
            if name.startswith("W") or name.startswith("b"):
                v = getattr(self, name)
                if isinstance(v, np.ndarray):
                    setattr(self, name, np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32))
        self.reward_list.append(float(np.mean(R)))
        self.epsilon = min(self.epsilon + (self.epsilon_increment or 0.0), self.epsilon_max)
        self.learn_step_counter += 1

    def copy_from(self, other, copy_optimizer_state=False):
        if other is None or getattr(other, "n_features", None) != self.n_features or getattr(other, "n_actions", None) != self.n_actions:
            return False
        self.W1 = other.W1.copy(); self.b1 = other.b1.copy(); self.tW1 = other.tW1.copy(); self.tb1 = other.tb1.copy()
        if self.dueling and getattr(other, "dueling", False):
            self.Wv = other.Wv.copy(); self.bv = other.bv.copy(); self.Wa = other.Wa.copy(); self.ba = other.ba.copy()
        elif self.dueling and hasattr(other, "W2"):
            self.Wv = np.zeros_like(self.Wv); self.bv = np.zeros_like(self.bv)
            self.Wa = other.W2.copy(); self.ba = other.b2.copy()
        elif (not self.dueling) and getattr(other, "dueling", False):
            self.W2 = other.Wa.copy(); self.b2 = other.ba.copy()
        elif hasattr(other, "W2"):
            self.W2 = other.W2.copy(); self.b2 = other.b2.copy()
        if copy_optimizer_state:
            self.epsilon = float(getattr(other, "epsilon", self.epsilon))
        self._copy_eval_to_target()
        return True

    def set_epsilon(self, value):
        self.epsilon = float(np.clip(value, 0.0, self.epsilon_max))

    def save_model(self, path, metadata=None):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        payload = {"scope": np.array(self.scope), "n_actions": self.n_actions, "n_features": self.n_features,
                   "W1": self.W1, "b1": self.b1, "epsilon": self.epsilon,
                   "reward_list": np.asarray(self.reward_list, dtype=np.float32), "metadata": np.array(str(metadata or {}))}
        if self.dueling:
            payload.update({"Wv": self.Wv, "bv": self.bv, "Wa": self.Wa, "ba": self.ba})
        else:
            payload.update({"W2": self.W2, "b2": self.b2})
        np.savez_compressed(path, **payload)
        return path

    def load_model(self, path, strict=False, load_epsilon=True):
        """Load a lightweight .npz checkpoint saved by save_model.

        strict=False lets related heads be reused across DQN/dueling variants:
        a plain DQN head can initialize a dueling actor advantage head, and a
        dueling advantage head can initialize a plain DQN head.
        """
        data = np.load(path, allow_pickle=True)
        n_actions = int(data["n_actions"]) if "n_actions" in data else self.n_actions
        n_features = int(data["n_features"]) if "n_features" in data else self.n_features
        if n_actions != self.n_actions or n_features != self.n_features:
            msg = (f"checkpoint shape mismatch for {self.scope}: "
                   f"ckpt=({n_features},{n_actions}) model=({self.n_features},{self.n_actions})")
            if strict:
                raise ValueError(msg)
            print(f"[Weight load warning] {msg}; skipped {path}")
            return False
        self.W1 = np.asarray(data["W1"], dtype=np.float32).copy()
        self.b1 = np.asarray(data["b1"], dtype=np.float32).copy()
        if self.dueling:
            if all(k in data for k in ["Wv", "bv", "Wa", "ba"]):
                self.Wv = np.asarray(data["Wv"], dtype=np.float32).copy()
                self.bv = np.asarray(data["bv"], dtype=np.float32).copy()
                self.Wa = np.asarray(data["Wa"], dtype=np.float32).copy()
                self.ba = np.asarray(data["ba"], dtype=np.float32).copy()
            elif all(k in data for k in ["W2", "b2"]):
                self.Wv = np.zeros_like(self.Wv, dtype=np.float32)
                self.bv = np.zeros_like(self.bv, dtype=np.float32)
                self.Wa = np.asarray(data["W2"], dtype=np.float32).copy()
                self.ba = np.asarray(data["b2"], dtype=np.float32).copy()
            else:
                raise ValueError(f"Unsupported DQN checkpoint format: {path}")
        else:
            if all(k in data for k in ["W2", "b2"]):
                self.W2 = np.asarray(data["W2"], dtype=np.float32).copy()
                self.b2 = np.asarray(data["b2"], dtype=np.float32).copy()
            elif all(k in data for k in ["Wa", "ba"]):
                self.W2 = np.asarray(data["Wa"], dtype=np.float32).copy()
                self.b2 = np.asarray(data["ba"], dtype=np.float32).copy()
            else:
                raise ValueError(f"Unsupported DQN checkpoint format: {path}")
        if load_epsilon and "epsilon" in data:
            self.set_epsilon(float(np.asarray(data["epsilon"]).item()))
        self._copy_eval_to_target()
        return True


class DuelingDoubleDQN(baseline_DQN):
    def __init__(self, n_actions, n_features, hidden_units=64, scope="RA_DDQN", learning_rate=0.002, **kwargs):
        super().__init__(n_actions=n_actions, n_features=n_features, hidden_units=hidden_units,
                         scope=scope, double_dqn=True, dueling=True, learning_rate=learning_rate, **kwargs)


class OptionActorCritic:
    """Masked actor-critic used for HCRL mode, primary and backup policies."""
    def __init__(self, n_actions, n_features, learning_rate=0.001, critic_lr=None, reward_decay=0.9,
                 entropy_coef=0.01, value_coef=0.5, memory_size=5000, batch_size=64,
                 hidden_units=64, scope="OptionAC", reward_clip=3.0, grad_clip=5.0, seed=6):
        self.n_actions = int(n_actions); self.n_features = int(n_features)
        self.lr = float(learning_rate); self.critic_lr = float(critic_lr or learning_rate)
        self.gamma = float(reward_decay); self.entropy_coef = float(entropy_coef); self.value_coef = float(value_coef)
        self.memory_size = int(memory_size); self.batch_size = int(batch_size); self.hidden_units = int(hidden_units)
        self.scope = str(scope); self.reward_clip = float(reward_clip); self.grad_clip = float(grad_clip)
        self.rng = np.random.RandomState(seed + sum(ord(c) for c in self.scope) % 10000)
        self.replay_buffer = deque(maxlen=self.memory_size); self.reward_list = []; self.learn_step_counter = 0; self.temperature = 1.0
        self._init_params()

    def _init_params(self):
        lim = np.sqrt(6.0 / (self.n_features + self.hidden_units))
        self.W1 = self.rng.uniform(-lim, lim, (self.n_features, self.hidden_units)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_units, dtype=np.float32)
        self.Wp = self.rng.uniform(-0.1, 0.1, (self.hidden_units, self.n_actions)).astype(np.float32)
        self.bp = np.zeros(self.n_actions, dtype=np.float32)
        self.Wv = self.rng.uniform(-0.1, 0.1, (self.hidden_units, 1)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)

    def _sanitize_state(self, state):
        x = np.asarray(state, dtype=np.float32); x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(x / max(float(np.max(np.abs(x))), 1.0), -10, 10).astype(np.float32)

    def _sanitize_mask(self, mask=None):
        if mask is None:
            return np.ones(self.n_actions, dtype=bool)
        m = np.asarray(mask, dtype=bool)
        if m.shape[0] != self.n_actions or not np.any(m):
            return np.ones(self.n_actions, dtype=bool)
        return m

    def _forward(self, X, return_cache=False):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1: X = X[None, :]
        Z = X.dot(self.W1) + self.b1; H = np.maximum(Z, 0.0)
        logits = np.nan_to_num(H.dot(self.Wp) + self.bp, nan=0.0, posinf=30.0, neginf=-30.0)
        value = np.nan_to_num((H.dot(self.Wv) + self.bv).squeeze(-1), nan=0.0, posinf=1e3, neginf=-1e3)
        return (logits, value, (X, Z, H)) if return_cache else (logits, value)

    def _softmax(self, logits, mask=None):
        mask = self._sanitize_mask(mask)
        z = np.asarray(logits, dtype=np.float64) / max(self.temperature, 1e-6)
        z = np.where(mask, np.clip(z, -30, 30), -1e9); z -= np.max(z)
        e = np.exp(z) * mask.astype(np.float64); s = np.sum(e)
        if s <= 1e-12 or not np.isfinite(s):
            p = np.zeros(self.n_actions, dtype=np.float64); p[np.where(mask)[0]] = 1.0 / np.sum(mask); return p
        p = e / s; return p / np.sum(p)

    def choose_action(self, observation, action_mask=None):
        x = self._sanitize_state(observation); logits, _ = self._forward(x)
        return int(self.rng.choice(self.n_actions, p=self._softmax(logits[0], action_mask)))

    def choose_best_action(self, observation, action_mask=None):
        x = self._sanitize_state(observation); logits, _ = self._forward(x); mask = self._sanitize_mask(action_mask)
        z = np.full(self.n_actions, -1e9, dtype=np.float32); z[mask] = logits[0][mask]
        return int(np.argmax(z))

    def store_transition(self, s, a, r, s_, next_action_mask=None, action_mask=None):
        self.replay_buffer.append((self._sanitize_state(s), int(a), float(np.clip(r, -self.reward_clip, self.reward_clip)),
                                   self._sanitize_state(s_), self._sanitize_mask(next_action_mask), self._sanitize_mask(action_mask)))

    def _clip(self, g):
        g = np.nan_to_num(g, nan=0.0, posinf=self.grad_clip, neginf=-self.grad_clip); n = np.linalg.norm(g)
        return g * (self.grad_clip / (n + 1e-8)) if n > self.grad_clip else g

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        S = np.asarray([b[0] for b in batch], dtype=np.float32); A = np.asarray([b[1] for b in batch], dtype=np.int32)
        R = np.asarray([b[2] for b in batch], dtype=np.float32); S2 = np.asarray([b[3] for b in batch], dtype=np.float32)
        next_masks = np.asarray([b[4] for b in batch], dtype=bool); masks = np.asarray([b[5] for b in batch], dtype=bool)
        logits, values, cache = self._forward(S, return_cache=True); _, next_values = self._forward(S2)
        targets = np.clip(R + self.gamma * next_values, -self.reward_clip, self.reward_clip)
        adv = targets - values; adv = np.clip((adv - np.mean(adv)) / (np.std(adv) + 1e-8), -5, 5).astype(np.float32)
        probs = np.vstack([self._softmax(l, m) for l, m in zip(logits, masks)]).astype(np.float32)
        one = np.zeros_like(probs); one[np.arange(self.batch_size), A] = 1.0
        valid_counts = np.maximum(np.sum(masks, axis=1, keepdims=True), 1)
        uniform = masks.astype(np.float32) / valid_counts
        dlogits = ((one - probs) * adv[:, None] + self.entropy_coef * (uniform - probs)) / self.batch_size
        dlogits *= masks.astype(np.float32)
        X, Z, H = cache
        gWp = H.T.dot(dlogits); gbp = np.sum(dlogits, axis=0)
        dvalue = ((values - targets) / self.batch_size).astype(np.float32)[:, None]
        gWv = H.T.dot(dvalue); gbv = np.sum(dvalue, axis=0)
        dH = dlogits.dot(self.Wp.T) - self.value_coef * dvalue.dot(self.Wv.T)
        dZ = dH * (Z > 0); gW1 = X.T.dot(dZ); gb1 = np.sum(dZ, axis=0)
        self.W1 += self.lr * self._clip(gW1).astype(np.float32); self.b1 += self.lr * self._clip(gb1).astype(np.float32)
        self.Wp += self.lr * self._clip(gWp).astype(np.float32); self.bp += self.lr * self._clip(gbp).astype(np.float32)
        self.Wv -= self.critic_lr * self._clip(gWv).astype(np.float32); self.bv -= self.critic_lr * self._clip(gbv).astype(np.float32)
        for n in ["W1", "b1", "Wp", "bp", "Wv", "bv"]:
            setattr(self, n, np.nan_to_num(getattr(self, n), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32))
        self.reward_list.append(float(np.mean(R))); self.learn_step_counter += 1

    def copy_from(self, other, copy_optimizer_state=False):
        if other is None or getattr(other, "n_features", None) != self.n_features or getattr(other, "n_actions", None) != self.n_actions:
            return False
        self.W1 = other.W1.copy(); self.b1 = other.b1.copy()
        if getattr(other, "dueling", False):
            self.Wp = other.Wa.copy(); self.bp = other.ba.copy()
        elif hasattr(other, "W2"):
            self.Wp = other.W2.copy(); self.bp = other.b2.copy()
        else:
            return False
        self.Wv = np.zeros_like(self.Wv); self.bv = np.zeros_like(self.bv)
        return True

    def set_epsilon(self, value):
        self.temperature = float(np.clip(1.0 - 0.5 * value, 0.35, 1.0))

    def save_model(self, path, metadata=None):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        np.savez_compressed(path, model_type=np.array("OptionActorCritic"), scope=np.array(self.scope),
                            n_actions=self.n_actions, n_features=self.n_features, W1=self.W1, b1=self.b1,
                            Wp=self.Wp, bp=self.bp, Wv=self.Wv, bv=self.bv,
                            reward_list=np.asarray(self.reward_list, dtype=np.float32), metadata=np.array(str(metadata or {})))
        return path

    def load_model(self, path, strict=False, load_epsilon=True):
        """Load a lightweight .npz checkpoint saved by OptionActorCritic or a compatible DQN head."""
        data = np.load(path, allow_pickle=True)
        n_actions = int(data["n_actions"]) if "n_actions" in data else self.n_actions
        n_features = int(data["n_features"]) if "n_features" in data else self.n_features
        if n_actions != self.n_actions or n_features != self.n_features:
            msg = (f"checkpoint shape mismatch for {self.scope}: "
                   f"ckpt=({n_features},{n_actions}) model=({self.n_features},{self.n_actions})")
            if strict:
                raise ValueError(msg)
            print(f"[Weight load warning] {msg}; skipped {path}")
            return False
        self.W1 = np.asarray(data["W1"], dtype=np.float32).copy()
        self.b1 = np.asarray(data["b1"], dtype=np.float32).copy()
        if all(k in data for k in ["Wp", "bp", "Wv", "bv"]):
            self.Wp = np.asarray(data["Wp"], dtype=np.float32).copy()
            self.bp = np.asarray(data["bp"], dtype=np.float32).copy()
            self.Wv = np.asarray(data["Wv"], dtype=np.float32).copy()
            self.bv = np.asarray(data["bv"], dtype=np.float32).copy()
        elif all(k in data for k in ["W2", "b2"]):
            self.Wp = np.asarray(data["W2"], dtype=np.float32).copy()
            self.bp = np.asarray(data["b2"], dtype=np.float32).copy()
            self.Wv = np.zeros_like(self.Wv, dtype=np.float32)
            self.bv = np.zeros_like(self.bv, dtype=np.float32)
        elif all(k in data for k in ["Wa", "ba"]):
            self.Wp = np.asarray(data["Wa"], dtype=np.float32).copy()
            self.bp = np.asarray(data["ba"], dtype=np.float32).copy()
            self.Wv = np.zeros_like(self.Wv, dtype=np.float32)
            self.bv = np.zeros_like(self.bv, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported OptionActorCritic checkpoint format: {path}")
        return True


class baseline_PPO(OptionActorCritic):
    def __init__(self, n_actions, n_features, actor_lr=0.0015, critic_lr=0.003, reward_decay=0.9,
                 batch_size=64, update_epochs=4, hidden_units=64, scope="PPO", seed=6, **kwargs):
        super().__init__(n_actions=n_actions, n_features=n_features, learning_rate=actor_lr,
                         critic_lr=critic_lr, reward_decay=reward_decay, batch_size=batch_size,
                         hidden_units=hidden_units, scope=scope, seed=seed, **kwargs)
        self.update_epochs = int(update_epochs)

    def choose_action(self, observation, action_mask=None):
        x = self._sanitize_state(observation); logits, _ = self._forward(x); probs = self._softmax(logits[0], action_mask)
        action = int(self.rng.choice(self.n_actions, p=probs)); return action, float(probs[action])

    def store_transition(self, s, a, r, prob=None, action_mask=None):
        # PPO branch in main stores one-step transitions without next-state; use current state as bootstrap placeholder.
        self.replay_buffer.append((self._sanitize_state(s), int(a), float(np.clip(r, -self.reward_clip, self.reward_clip)),
                                   self._sanitize_state(s), self._sanitize_mask(action_mask), self._sanitize_mask(action_mask)))


class BLOR:
    """Bayesian bandit oracle selector."""
    def __init__(self, success_num, failure_num, oracle_cost, seed=None):
        self.success_num = np.asarray(success_num, dtype=float)
        self.failure_num = np.asarray(failure_num, dtype=float)
        self.oracle_cost = np.asarray(oracle_cost, dtype=float)
        self.rng = np.random.RandomState(seed if seed is not None else 123)

    def get_oracles(self, success_num, failure_num, oracle_cost):
        alpha = np.asarray(success_num, dtype=float) + 1.0
        beta = np.asarray(failure_num, dtype=float) + 1.0
        samples = self.rng.beta(alpha, beta)
        cost_norm = np.asarray(oracle_cost, dtype=float) / max(float(np.max(oracle_cost)), 1e-8)
        return samples - 0.25 * cost_norm

    def choose_action(self, scores):
        return int(np.argmax(scores))


class baselines:
    def __init__(self, n_actions, oracle_types=None):
        self.n_actions = int(n_actions)
        self.oracle_types = np.asarray(oracle_types if oracle_types is not None else np.zeros(n_actions), dtype=int)

    def random_choose_action(self, action_mask=None):
        if action_mask is None:
            return int(np.random.randint(0, self.n_actions))
        valid = np.where(np.asarray(action_mask, dtype=bool))[0]
        return int(np.random.choice(valid if valid.size else np.arange(self.n_actions)))

    def RR_choose_action(self, request_c, action_mask=None):
        if action_mask is None:
            return int((int(request_c) - 1) % self.n_actions)
        valid = np.where(np.asarray(action_mask, dtype=bool))[0]
        if valid.size == 0:
            return int((int(request_c) - 1) % self.n_actions)
        return int(valid[(int(request_c) - 1) % valid.size])

    def early_choose_action(self, idle_times, action_mask=None):
        idle = np.asarray(idle_times, dtype=float).copy()
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool)
            idle[~mask] = np.inf
        if not np.isfinite(idle).any():
            return int(np.argmin(np.asarray(idle_times, dtype=float)))
        return int(np.argmin(idle))

    def PSG_choose_action(self, rewards, cost, action_mask=None):
        rewards = np.asarray(rewards, dtype=float)
        cost = np.asarray(cost, dtype=float)
        score = rewards - 0.25 * cost
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool)
            score[~mask] = -1e9
        return int(np.argmax(score))
