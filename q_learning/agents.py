from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
from collections import Counter

import numpy as np

class DDQN:
    """
    Implements Double Deep Q-Learning

    a_max = arg max a' Q(s', a'; theta)
    y = r + gamma * Q(s', a_max, theta-)
    L = (y - Q(s, a; theta)) ** 2
    """
    def __init__(self, main, target, env, 
            experience_replay, 
            batch_size,
            max_path_length,
            train_update_freq=4,
            target_update_freq=10000,
            epsilon=0.1,
            gamma=0.99):

        self.main = main
        self.target = target
        self.env = env
        self.experience_replay = experience_replay

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.train_update_freq = train_update_freq
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon
        self.gamma = gamma

        # TODO: allow continous action spaces?
        self.n_actions = env.action_space.n
        self.steps_taken = 0

    def run_episode(self):
        obs = self.env.reset()

        actions = Counter()
        terminal = False
        total_reward = 0
        t = 0

        while not terminal and t < self.max_path_length:

            obs, terminal, info = self.step(obs)

            total_reward += info['reward']
            actions[info['action']] += 1
            t += 1

            if t % 3 == 0: self.env.render()

        print('Action statistics:', actions)
        return total_reward

    def play(self, epsilon=0.05, render=True):
        obs = self.env.reset()

        actions = Counter()
        terminal = False
        total_reward = 0
        t = 0

        while not terminal and t < self.max_path_length:
            if np.random.rand() < epsilon:
                action = np.random.choice(self.n_actions)
            else:
                qvals = self.main.predict(obs, batch_size=1)
                action = np.argmax(qvals)
            next_obs, reward, terminal, _ = self.env.step(action)

            total_reward += reward
            actions[action] += 1

            if t % 3 == 0: self.env.render()

        print('Action statistics:', actions)

    def step(self, obs):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            qvals = self.main.predict(obs, batch_size=1)
            action = np.argmax(qvals)
        next_obs, reward, terminal, _ = self.env.step(action)

        self.experience_replay.add((obs, action, reward, next_obs, terminal))
        obs = next_obs

        if self.steps_taken % self.train_update_freq == 0:

            b_obs, b_action, b_reward, b_next_obs, b_terminal = self.experience_replay.sample()
            next_main_qvals = self.main.predict_on_batch(b_next_obs)
            next_target_qvals = self.target.predict_on_batch(b_next_obs)
            a_max = np.argmax(next_main_qvals, axis=1)

            y = self.main.predict_on_batch(b_obs)
            y[range(self.batch_size), b_action] = b_reward + ~b_terminal * (self.gamma * next_target_qvals[range(self.batch_size), a_max])

            self.main.train_on_batch(b_obs, y)

        # Updates
        self.steps_taken += 1
        if self.steps_taken % self.target_update_freq == 0:
            print("Updating target weights, steps taken", self.steps_taken)
            self.target.set_weights(self.main.get_weights())

        return obs, terminal, dict(action=action, reward=reward)
