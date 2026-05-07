import os
# Keep TensorFlow quiet and force CPU mode unless the user explicitly changes this file.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings
warnings.filterwarnings("ignore", message="`tf.layers.dense` is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import random
from collections import deque

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

random.seed(6)
np.random.seed(6)
tf.set_random_seed(6)


class baseline_DQN:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
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
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max
        self.hidden_units = hidden_units
        self.scope = scope
        self.double_dqn = double_dqn
        self.dueling = dueling

        self.learn_step_counter = 0
        self.replay_buffer = deque(maxlen=self.memory_size)
        self.reward_list = []

        self._build_net()

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"{self.scope}/eval_net")
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"{self.scope}/target_net")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _q_head(self, x, scope, trainable=True):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(x, self.hidden_units, activation=tf.nn.relu, name="l1", trainable=trainable)
            l2 = tf.layers.dense(l1, self.hidden_units, activation=tf.nn.relu, name="l2", trainable=trainable)
            if self.dueling:
                value = tf.layers.dense(l2, 1, name="value", trainable=trainable)
                advantage = tf.layers.dense(l2, self.n_actions, name="advantage", trainable=trainable)
                q = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
            else:
                q = tf.layers.dense(l2, self.n_actions, name="q", trainable=trainable)
            return q

    def _build_net(self):
        with tf.variable_scope(self.scope):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name="s_")
            self.action_input = tf.placeholder(tf.int32, [None], name="action_input")
            self.q_target = tf.placeholder(tf.float32, [None], name="q_target")

            self.q_eval = self._q_head(self.s, "eval_net", trainable=True)
            self.q_next = self._q_head(self.s_, "target_net", trainable=False)

            action_one_hot = tf.one_hot(self.action_input, self.n_actions, dtype=tf.float32)
            q_eval_wrt_a = tf.reduce_sum(self.q_eval * action_one_hot, axis=1)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, q_eval_wrt_a))
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        self.replay_buffer.append((np.array(s, dtype=np.float32), int(a), float(r), np.array(s_, dtype=np.float32)))

    def choose_action(self, observation):
        observation = np.array(observation, dtype=np.float32)
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation[np.newaxis, :]})
            action = int(np.argmax(actions_value))
        else:
            action = int(np.random.randint(0, self.n_actions))
        return action

    def choose_best_action(self, observation):
        observation = np.array(observation, dtype=np.float32)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation[np.newaxis, :]})
        return int(np.argmax(actions_value))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in minibatch], dtype=np.float32)
        action_batch = np.array([data[1] for data in minibatch], dtype=np.int32)
        reward_batch = np.array([data[2] for data in minibatch], dtype=np.float32)
        next_state_batch = np.array([data[3] for data in minibatch], dtype=np.float32)

        q_next_batch = self.sess.run(self.q_next, feed_dict={self.s_: next_state_batch})
        if self.double_dqn:
            q_eval_next_batch = self.sess.run(self.q_eval, feed_dict={self.s: next_state_batch})
            best_actions = np.argmax(q_eval_next_batch, axis=1)
            q_target = reward_batch + self.gamma * q_next_batch[np.arange(self.batch_size), best_actions]
        else:
            q_target = reward_batch + self.gamma * np.max(q_next_batch, axis=1)

        self.sess.run(self._train_op, feed_dict={self.s: state_batch, self.action_input: action_batch, self.q_target: q_target})
        self.reward_list.append(float(np.mean(reward_batch)))

        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        self.learn_step_counter += 1


class DuelingDoubleDQN(baseline_DQN):
    def __init__(self, n_actions, n_features, hidden_units=64, scope="RA_DDQN", **kwargs):
        super().__init__(
            n_actions=n_actions,
            n_features=n_features,
            hidden_units=hidden_units,
            scope=scope,
            double_dqn=True,
            dueling=True,
            **kwargs,
        )


class baseline_PPO:
    def __init__(
        self,
        n_actions,
        n_features,
        actor_lr=0.001,
        critic_lr=0.002,
        reward_decay=0.9,
        clip_ratio=0.2,
        batch_size=64,
        update_epochs=5,
        entropy_coef=0.01,
        hidden_units=64,
        scope="PPO",
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = reward_decay
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self.hidden_units = hidden_units
        self.scope = scope

        self.states = []
        self.actions = []
        self.rewards = []
        self.old_action_probs = []
        self.reward_list = []

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.variable_scope(self.scope):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")
            self.a = tf.placeholder(tf.int32, [None], name="a")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.ret = tf.placeholder(tf.float32, [None], name="ret")
            self.old_prob = tf.placeholder(tf.float32, [None], name="old_prob")

            with tf.variable_scope("actor"):
                l1 = tf.layers.dense(self.s, self.hidden_units, activation=tf.nn.relu, name="l1")
                l2 = tf.layers.dense(l1, self.hidden_units, activation=tf.nn.relu, name="l2")
                logits = tf.layers.dense(l2, self.n_actions, name="logits")
                self.action_probs = tf.nn.softmax(logits)
                action_one_hot = tf.one_hot(self.a, self.n_actions)
                action_prob = tf.reduce_sum(self.action_probs * action_one_hot, axis=1)
                ratio = action_prob / (self.old_prob + 1e-8)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                entropy = -tf.reduce_sum(self.action_probs * tf.log(self.action_probs + 1e-8), axis=1)
                surrogate1 = ratio * self.adv
                surrogate2 = clipped_ratio * self.adv
                self.actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2) + self.entropy_coef * entropy)
                actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"{self.scope}/actor")
                self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss, var_list=actor_vars)

            with tf.variable_scope("critic"):
                c1 = tf.layers.dense(self.s, self.hidden_units, activation=tf.nn.relu, name="c1")
                c2 = tf.layers.dense(c1, self.hidden_units, activation=tf.nn.relu, name="c2")
                self.value = tf.squeeze(tf.layers.dense(c2, 1, name="value"), axis=1)
                self.critic_loss = tf.reduce_mean(tf.square(self.ret - self.value))
                critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f"{self.scope}/critic")
                self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss, var_list=critic_vars)

    def choose_action(self, state):
        state = np.array(state, dtype=np.float32)
        probs = self.sess.run(self.action_probs, feed_dict={self.s: state[np.newaxis, :]})[0]
        probs = np.nan_to_num(probs, nan=1.0 / self.n_actions, posinf=1.0 / self.n_actions, neginf=1.0 / self.n_actions)
        probs = probs / np.sum(probs)
        action = int(np.random.choice(self.n_actions, p=probs))
        return action, float(probs[action])

    def choose_best_action(self, state):
        state = np.array(state, dtype=np.float32)
        probs = self.sess.run(self.action_probs, feed_dict={self.s: state[np.newaxis, :]})[0]
        return int(np.argmax(probs))

    def store_transition(self, s, a, r, old_action_prob):
        self.states.append(np.array(s, dtype=np.float32))
        self.actions.append(int(a))
        self.rewards.append(float(r))
        self.old_action_probs.append(float(old_action_prob))

    def _discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            discounted[t] = running_return
        return discounted

    def learn(self):
        if len(self.states) < self.batch_size:
            return
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        rewards = np.array(self.rewards, dtype=np.float32)
        old_probs = np.array(self.old_action_probs, dtype=np.float32)

        returns = self._discount_rewards(rewards)
        values = self.sess.run(self.value, feed_dict={self.s: states})
        advantages = returns - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        for _ in range(self.update_epochs):
            self.sess.run(
                [self.actor_train_op, self.critic_train_op],
                feed_dict={
                    self.s: states,
                    self.a: actions,
                    self.adv: advantages,
                    self.ret: returns,
                    self.old_prob: old_probs,
                },
            )
        self.reward_list.append(float(np.mean(rewards)))
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.old_action_probs.clear()


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
