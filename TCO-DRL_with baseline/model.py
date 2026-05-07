import os
# Pure NumPy model implementations. No TensorFlow import is used, which avoids
# Windows native TensorFlow access-violation crashes such as exit code -1073741819.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import random
from collections import deque

import numpy as np

random.seed(6)
np.random.seed(6)


class baseline_DQN:
    """Crash-safe NumPy DQN baseline.

    This class keeps the public API of the original TensorFlow DQN used by main.py:
      - choose_action(observation, action_mask=None)
      - choose_best_action(observation, action_mask=None)
      - store_transition(s, a, r, s_, next_action_mask=None)
      - learn()

    It replaces TensorFlow with a small NumPy MLP to avoid native crashes on
    CPU-only Windows environments. It supports Double DQN and Dueling heads for
    RA-DDQN while staying deterministic and lightweight.
    """

    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.003,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=800,
        batch_size=30,
        e_greedy_increment=0.001,
        hidden_units=64,
        scope="DQN",
        double_dqn=False,
        dueling=False,
        reward_clip=20.0,
        grad_clip=5.0,
        seed=6,
    ):
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
        self.scope = scope
        self.double_dqn = bool(double_dqn)
        self.dueling = bool(dueling)
        self.reward_clip = float(reward_clip)
        self.grad_clip = float(grad_clip)

        # Use a different deterministic seed per scope, so DQN and RA-DDQN are not identical.
        scope_offset = sum(ord(c) for c in str(scope)) % 10000
        self.rng = np.random.RandomState(seed + scope_offset)

        self.learn_step_counter = 0
        self.replay_buffer = deque(maxlen=self.memory_size)
        self.reward_list = []

        self._init_params()
        self._copy_eval_to_target()

    def _init_params(self):
        # Xavier-like small initialization. States are normalized before use.
        limit1 = np.sqrt(6.0 / (self.n_features + self.hidden_units))
        self.W1 = self.rng.uniform(-limit1, limit1, (self.n_features, self.hidden_units)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_units, dtype=np.float32)

        if self.dueling:
            limit_v = np.sqrt(6.0 / (self.hidden_units + 1))
            limit_a = np.sqrt(6.0 / (self.hidden_units + self.n_actions))
            self.Wv = self.rng.uniform(-limit_v, limit_v, (self.hidden_units, 1)).astype(np.float32)
            self.bv = np.zeros(1, dtype=np.float32)
            self.Wa = self.rng.uniform(-limit_a, limit_a, (self.hidden_units, self.n_actions)).astype(np.float32)
            self.ba = np.zeros(self.n_actions, dtype=np.float32)
        else:
            limit2 = np.sqrt(6.0 / (self.hidden_units + self.n_actions))
            self.W2 = self.rng.uniform(-limit2, limit2, (self.hidden_units, self.n_actions)).astype(np.float32)
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
        # Per-state normalization keeps arrival/wait-time scale from dominating.
        scale = max(float(np.max(np.abs(x))), 1.0)
        x = np.clip(x / scale, -10.0, 10.0).astype(np.float32)
        return x

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
        if target:
            W1, b1 = self.tW1, self.tb1
        else:
            W1, b1 = self.W1, self.b1
        Z1 = X.dot(W1) + b1
        H = np.maximum(Z1, 0.0)
        if self.dueling:
            if target:
                V = H.dot(self.tWv) + self.tbv
                A = H.dot(self.tWa) + self.tba
            else:
                V = H.dot(self.Wv) + self.bv
                A = H.dot(self.Wa) + self.ba
            Q = V + (A - np.mean(A, axis=1, keepdims=True))
        else:
            if target:
                Q = H.dot(self.tW2) + self.tb2
            else:
                Q = H.dot(self.W2) + self.b2
        Q = np.nan_to_num(Q, nan=0.0, posinf=1e3, neginf=-1e3)
        if return_cache:
            return Q, (X, Z1, H)
        return Q

    def store_transition(self, s, a, r, s_, next_action_mask=None):
        """Store one transition.

        next_action_mask is important when Action_Mask_Mode='type'. The agent
        selects only valid same-type oracle actions at execution time, so the
        Bellman target must also maximize only over valid actions in the next
        state. Without this, Q-learning can overestimate invalid actions that
        will never be executable.
        """
        self.replay_buffer.append((
            self._sanitize_state(s),
            int(a),
            float(np.clip(r, -self.reward_clip, self.reward_clip)),
            self._sanitize_state(s_),
            self._sanitize_mask(next_action_mask),
        ))

    def choose_action(self, observation, action_mask=None):
        x = self._sanitize_state(observation)
        valid_actions = np.where(self._sanitize_mask(action_mask))[0]
        if self.rng.uniform() < self.epsilon:
            q = self._forward(x)[0]
            masked_q = np.full(self.n_actions, -1e9, dtype=np.float32)
            masked_q[valid_actions] = q[valid_actions]
            return int(np.argmax(masked_q))
        return int(self.rng.choice(valid_actions))

    def choose_best_action(self, observation, action_mask=None):
        x = self._sanitize_state(observation)
        q = self._forward(x)[0]
        mask = self._sanitize_mask(action_mask)
        valid_actions = np.where(mask)[0]
        masked_q = np.full(self.n_actions, -1e9, dtype=np.float32)
        masked_q[valid_actions] = q[valid_actions]
        return int(np.argmax(masked_q))

    def _clip_grad(self, g):
        g = np.nan_to_num(g, nan=0.0, posinf=self.grad_clip, neginf=-self.grad_clip)
        norm = np.linalg.norm(g)
        if norm > self.grad_clip:
            g = g * (self.grad_clip / (norm + 1e-8))
        return g

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        if self.learn_step_counter % self.replace_target_iter == 0:
            self._copy_eval_to_target()

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        S = np.asarray([d[0] for d in minibatch], dtype=np.float32)
        A = np.asarray([d[1] for d in minibatch], dtype=np.int32)
        R = np.asarray([d[2] for d in minibatch], dtype=np.float32)
        S2 = np.asarray([d[3] for d in minibatch], dtype=np.float32)
        next_masks = np.asarray([d[4] for d in minibatch], dtype=bool)

        q_next_t = self._forward(S2, target=True)
        # Apply the next-state valid-action mask to the Bellman target. This
        # makes training consistent with masked execution in choose_action().
        masked_q_next_t = np.where(next_masks, q_next_t, -1e9)
        if self.double_dqn:
            q_next_eval = self._forward(S2, target=False)
            masked_q_next_eval = np.where(next_masks, q_next_eval, -1e9)
            best_next = np.argmax(masked_q_next_eval, axis=1)
            y = R + self.gamma * q_next_t[np.arange(self.batch_size), best_next]
        else:
            y = R + self.gamma * np.max(masked_q_next_t, axis=1)
        y = np.clip(np.nan_to_num(y, nan=0.0, posinf=self.reward_clip, neginf=-self.reward_clip), -self.reward_clip, self.reward_clip)

        q, cache = self._forward(S, target=False, return_cache=True)
        X, Z1, H = cache
        pred = q[np.arange(self.batch_size), A]
        err = np.clip((pred - y) / self.batch_size, -5.0, 5.0).astype(np.float32)

        # Backprop through selected Q values.
        dQ = np.zeros_like(q, dtype=np.float32)
        dQ[np.arange(self.batch_size), A] = err

        if self.dueling:
            # Q = V + A - mean(A). dV sums all action gradients; dA subtracts mean component.
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

        # Clip gradients and update.
        self.W1 -= self.lr * self._clip_grad(gW1).astype(np.float32)
        self.b1 -= self.lr * self._clip_grad(gb1).astype(np.float32)
        if self.dueling:
            self.Wv -= self.lr * self._clip_grad(gWv).astype(np.float32)
            self.bv -= self.lr * self._clip_grad(gbv).astype(np.float32)
            self.Wa -= self.lr * self._clip_grad(gWa).astype(np.float32)
            self.ba -= self.lr * self._clip_grad(gba).astype(np.float32)
        else:
            self.W2 -= self.lr * self._clip_grad(gW2).astype(np.float32)
            self.b2 -= self.lr * self._clip_grad(gb2).astype(np.float32)

        # Final safety against NaNs.
        for name in ["W1", "b1"]:
            setattr(self, name, np.nan_to_num(getattr(self, name), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32))
        if self.dueling:
            for name in ["Wv", "bv", "Wa", "ba"]:
                setattr(self, name, np.nan_to_num(getattr(self, name), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32))
        else:
            for name in ["W2", "b2"]:
                setattr(self, name, np.nan_to_num(getattr(self, name), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32))

        self.reward_list.append(float(np.mean(R)))
        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        self.learn_step_counter += 1


class DuelingDoubleDQN(baseline_DQN):
    def __init__(self, n_actions, n_features, hidden_units=64, scope="RA_DDQN", learning_rate=0.002, **kwargs):
        super().__init__(
            n_actions=n_actions,
            n_features=n_features,
            hidden_units=hidden_units,
            scope=scope,
            double_dqn=True,
            dueling=True,
            learning_rate=learning_rate,
            **kwargs,
        )


class baseline_PPO:
    """Crash-safe PPO-style baseline implemented in NumPy.

    The previous TensorFlow PPO graph could crash the Python process on Windows
    under the rl_hard scenario when very large/negative rewards made the policy
    update numerically unstable. This class keeps the same public API used by
    main.py, but avoids a second TensorFlow Session and adds:
      - reward clipping and return normalization
      - action-mask-aware sampling and learning
      - probability clipping to avoid log/ratio overflow
      - gradient clipping

    It is intentionally lightweight: a linear softmax actor plus a linear critic.
    It is meant as a stable RL baseline, while DQN/RA-DDQN remain the main methods.
    """

    def __init__(
        self,
        n_actions,
        n_features,
        actor_lr=0.002,
        critic_lr=0.005,
        reward_decay=0.90,
        clip_ratio=0.20,
        batch_size=64,
        update_epochs=4,
        entropy_coef=0.01,
        hidden_units=64,  # kept for CLI/API compatibility
        scope="PPO",
        reward_clip=10.0,
        grad_clip=1.0,
        seed=6,
    ):
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.gamma = float(reward_decay)
        self.clip_ratio = float(clip_ratio)
        self.batch_size = int(batch_size)
        self.update_epochs = int(update_epochs)
        self.entropy_coef = float(entropy_coef)
        self.hidden_units = hidden_units
        self.scope = scope
        self.reward_clip = float(reward_clip)
        self.grad_clip = float(grad_clip)

        self.rng = np.random.RandomState(seed)
        # Small initialization is important because state dimensionality grows with oracle count.
        self.actor_w = self.rng.normal(0.0, 0.01, size=(self.n_features, self.n_actions)).astype(np.float32)
        self.actor_b = np.zeros(self.n_actions, dtype=np.float32)
        self.critic_w = self.rng.normal(0.0, 0.01, size=(self.n_features,)).astype(np.float32)
        self.critic_b = np.float32(0.0)

        self.states = []
        self.actions = []
        self.rewards = []
        self.old_action_probs = []
        self.masks = []
        self.reward_list = []

    def _sanitize_state(self, state):
        x = np.asarray(state, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        # Per-state scale normalization prevents huge time features from dominating.
        scale = np.maximum(np.max(np.abs(x)), 1.0)
        x = np.clip(x / scale, -10.0, 10.0)
        return x.astype(np.float32)

    def _sanitize_mask(self, action_mask=None):
        if action_mask is None:
            return np.ones(self.n_actions, dtype=bool)
        mask = np.asarray(action_mask, dtype=bool)
        if mask.shape[0] != self.n_actions or not np.any(mask):
            return np.ones(self.n_actions, dtype=bool)
        return mask

    def _softmax(self, logits, mask=None):
        logits = np.asarray(logits, dtype=np.float64)
        logits = np.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = np.clip(logits, -30.0, 30.0)
        if mask is not None:
            mask = self._sanitize_mask(mask)
            logits = np.where(mask, logits, -1e9)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        if mask is not None:
            exp_logits = exp_logits * mask.astype(np.float64)
        denom = np.sum(exp_logits)
        if denom <= 1e-12 or not np.isfinite(denom):
            if mask is None:
                return np.ones(self.n_actions, dtype=np.float64) / self.n_actions
            valid = np.where(mask)[0]
            probs = np.zeros(self.n_actions, dtype=np.float64)
            probs[valid] = 1.0 / len(valid)
            return probs
        probs = exp_logits / denom
        probs = np.clip(probs, 1e-8, 1.0)
        if mask is not None:
            probs = probs * mask.astype(np.float64)
        probs = probs / np.sum(probs)
        return probs

    def _policy(self, states, masks=None):
        states = np.asarray(states, dtype=np.float32)
        logits = states.dot(self.actor_w) + self.actor_b
        if masks is None:
            return np.vstack([self._softmax(row) for row in logits])
        return np.vstack([self._softmax(row, mask) for row, mask in zip(logits, masks)])

    def _value(self, states):
        states = np.asarray(states, dtype=np.float32)
        return states.dot(self.critic_w) + self.critic_b

    def choose_action(self, state, action_mask=None):
        x = self._sanitize_state(state)
        mask = self._sanitize_mask(action_mask)
        probs = self._softmax(x.dot(self.actor_w) + self.actor_b, mask)
        action = int(self.rng.choice(self.n_actions, p=probs))
        return action, float(max(probs[action], 1e-8))

    def choose_best_action(self, state, action_mask=None):
        x = self._sanitize_state(state)
        mask = self._sanitize_mask(action_mask)
        probs = self._softmax(x.dot(self.actor_w) + self.actor_b, mask)
        return int(np.argmax(probs))

    def store_transition(self, s, a, r, old_action_prob, action_mask=None):
        self.states.append(self._sanitize_state(s))
        self.actions.append(int(a))
        # Clip rewards locally. This fixes the crash caused by extreme PPO returns,
        # while leaving the environment's printed rewards unchanged.
        self.rewards.append(float(np.clip(r, -self.reward_clip, self.reward_clip)))
        self.old_action_probs.append(float(np.clip(old_action_prob, 1e-6, 1.0)))
        self.masks.append(self._sanitize_mask(action_mask))

    def _discount_rewards(self, rewards):
        rewards = np.asarray(rewards, dtype=np.float32)
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = float(rewards[t]) + self.gamma * running_return
            discounted[t] = running_return
        return discounted

    def _clip_grad(self, grad):
        norm = np.linalg.norm(grad)
        if not np.isfinite(norm):
            return np.zeros_like(grad)
        if norm > self.grad_clip:
            grad = grad * (self.grad_clip / (norm + 1e-8))
        return grad

    def learn(self):
        if len(self.states) < self.batch_size:
            return

        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        old_probs = np.asarray(self.old_action_probs, dtype=np.float64)
        masks = np.asarray(self.masks, dtype=bool)

        returns = self._discount_rewards(rewards)
        returns = np.nan_to_num(returns, nan=0.0, posinf=self.reward_clip, neginf=-self.reward_clip)
        # Normalize returns for critic stability.
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = np.clip(returns, -5.0, 5.0).astype(np.float32)

        for _ in range(self.update_epochs):
            values = self._value(states).astype(np.float32)
            advantages = returns - values
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            advantages = np.clip(advantages, -5.0, 5.0)

            probs = self._policy(states, masks)
            selected_probs = np.clip(probs[np.arange(len(actions)), actions], 1e-8, 1.0)
            ratios = np.clip(selected_probs / old_probs, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            coeff = (ratios * advantages).astype(np.float32)

            one_hot = np.zeros_like(probs, dtype=np.float32)
            one_hot[np.arange(len(actions)), actions] = 1.0
            grad_logits = (coeff[:, None] * (one_hot - probs.astype(np.float32))) / max(len(actions), 1)
            # Do not update invalid actions for masked states.
            grad_logits *= masks.astype(np.float32)

            # Entropy bonus: push slightly away from deterministic collapse.
            if self.entropy_coef > 0:
                valid_counts = np.maximum(np.sum(masks, axis=1, keepdims=True), 1)
                uniform = masks.astype(np.float32) / valid_counts
                grad_logits += self.entropy_coef * (uniform - probs.astype(np.float32)) / max(len(actions), 1)

            grad_w = states.T.dot(grad_logits)
            grad_b = np.sum(grad_logits, axis=0)
            grad_w = self._clip_grad(grad_w)
            grad_b = self._clip_grad(grad_b)
            self.actor_w += self.actor_lr * grad_w.astype(np.float32)
            self.actor_b += self.actor_lr * grad_b.astype(np.float32)

            # Linear critic update: minimize 0.5 * (return - value)^2.
            values = self._value(states).astype(np.float32)
            error = np.clip(returns - values, -5.0, 5.0)
            grad_v = states.T.dot(error) / max(len(actions), 1)
            grad_vb = np.mean(error)
            grad_v = self._clip_grad(grad_v)
            self.critic_w += self.critic_lr * grad_v.astype(np.float32)
            self.critic_b = np.float32(self.critic_b + self.critic_lr * np.clip(grad_vb, -self.grad_clip, self.grad_clip))

            # Final safety: avoid silent NaN propagation.
            self.actor_w = np.nan_to_num(self.actor_w, nan=0.0, posinf=1.0, neginf=-1.0)
            self.actor_b = np.nan_to_num(self.actor_b, nan=0.0, posinf=1.0, neginf=-1.0)
            self.critic_w = np.nan_to_num(self.critic_w, nan=0.0, posinf=1.0, neginf=-1.0)
            self.critic_b = np.float32(np.nan_to_num(self.critic_b, nan=0.0, posinf=1.0, neginf=-1.0))

        self.reward_list.append(float(np.mean(rewards)))
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.old_action_probs.clear()
        self.masks.clear()


class baselines:
    def __init__(self, action_num, oracle_types):
        self.action_num = action_num
        self.oracle_types = np.array(oracle_types)

    def random_choose_action(self):
        return int(np.random.randint(0, self.action_num))

    def RR_choose_action(self, request_count):
        return int((request_count - 1) % self.action_num)

    def early_choose_action(self, idle_times):
        return int(np.argmin(idle_times))

    def PSG_choose_action(self, rewards, cost):
        rewards = np.array(rewards)
        cost = np.array(cost)
        candidate_idx = np.where(rewards > 0)[0]
        if len(candidate_idx) == 0:
            return int(np.argmax(rewards))
        candidate_rewards = rewards[candidate_idx]
        max_reward = np.max(candidate_rewards)
        # Keep high-reward candidates and choose the cheaper one.
        high = candidate_idx[candidate_rewards >= max_reward * 0.95]
        return int(high[np.argmin(cost[high])])


class BLOR:
    """A simple Bayesian bandit style baseline for oracle selection.

    It approximates BLOR behavior using Beta posterior sampling with a cost penalty.
    """

    def __init__(self, success_num, failure_num, oracle_cost, cost_weight=0.2):
        self.success_num = np.array(success_num, dtype=float)
        self.failure_num = np.array(failure_num, dtype=float)
        self.oracle_cost = np.array(oracle_cost, dtype=float)
        self.cost_weight = cost_weight

    def get_oracles(self, success_num=None, failure_num=None, oracle_cost=None):
        if success_num is not None:
            self.success_num = np.array(success_num, dtype=float)
        if failure_num is not None:
            self.failure_num = np.array(failure_num, dtype=float)
        if oracle_cost is not None:
            self.oracle_cost = np.array(oracle_cost, dtype=float)
        alpha = self.success_num + 1.0
        beta = self.failure_num + 1.0
        sampled_reliability = np.random.beta(alpha, beta)
        return sampled_reliability - self.cost_weight * self.oracle_cost

    def choose_action(self, oracle_scores):
        return int(np.argmax(oracle_scores))
