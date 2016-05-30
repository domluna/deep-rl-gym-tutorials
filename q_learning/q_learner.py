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
            observation_preprocess=None,
            batch_size=32,
            max_path_length=10000,
            target_update_freq=1000,
            epsilon=0.1,
            gamma=0.99):

        self.main = main
        self.target = target
        self.env = env
        self.experience_replay = experience_replay
        self.obs_preprocess = observation_preprocess

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon
        self.gamma = gamma

        # TODO: allow continous action spaces
        self.n_actions = env.action_space.n
        self.steps_taken = 0

    def run_episode(self, render=False):
        obs = self.env.reset()
        if self.obs_preprocess:
            obs = self.obs_preprocess(obs)

        terminal = False
        losses = []
        total_reward = 0
        t = 0
        actions = Counter()

        while not terminal and t < self.max_path_length:
            obs, terminal, info = self.step(obs)
            total_reward += info['reward']
            losses.append(info['loss'])
            actions[info['action']] += 1

            if render and t % 3 == 0: self.env.render()

            t += 1

        print('Action statistics:', actions)

        return losses, total_reward

    def step(self, obs):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            qvals = self.main.predict(obs, batch_size=1)
            action = np.argmax(qvals)
        next_obs, reward, terminal, _ = self.env.step(action)

        if self.obs_preprocess:
            next_obs = self.obs_preprocess(next_obs)

        self.experience_replay.add((obs, action, reward, next_obs, terminal))
        obs = next_obs

        b_obs, b_action, b_reward, b_next_obs, b_terminal = self.experience_replay.sample(self.batch_size)

        # DDQN loss calculation
        main_qvals = self.main.predict_on_batch(b_obs)
        next_main_qvals = self.main.predict_on_batch(b_next_obs)
        next_target_qvals = self.target.predict_on_batch(b_next_obs)
        a_max = np.argmax(next_main_qvals, axis=1)

        # print('main', main_qvals)
        # print('next main', next_main_qvals)
        # print('next target', next_target_qvals)

        y = b_reward + ~b_terminal * (self.gamma * next_target_qvals[range(self.batch_size), a_max])
        main_qvals[range(self.batch_size), b_action] = y

        # print('a max', a_max)
        # print('reward', b_reward)

        loss = self.main.train_on_batch(b_obs, main_qvals)
        # print('loss', loss)
        # print('**************************')

        # Updates
        self.steps_taken += 1
        if self.steps_taken % self.target_update_freq == 0:
            print("Updating target weights, self.steps_taken", self.steps_taken)
            self.target.set_weights(self.main.get_weights())

        return obs, terminal, dict(loss=loss, action=action, reward=reward)
