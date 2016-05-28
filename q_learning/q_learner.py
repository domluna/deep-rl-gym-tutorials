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

        done = False
        losses = []
        total_reward = 0
        t = 0
        actions = Counter()

        while not done and t < self.max_path_length:
            obs, done, info = self.run_step(obs)
            total_reward += info['reward']
            losses.append(info['loss'])
            actions[info['action']] += 1

            if render and t % 3 == 0: self.env.render()

            t += 1

        print('Action statistics:', actions)

        return losses, total_reward

    def run_step(self, obs):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            qvals = self.main.predict(obs, batch_size=1)
            action = np.argmax(qvals)
        obs_next, reward, done, _ = self.env.step(action)

        if self.obs_preprocess:
            obs_next = self.obs_preprocess(obs_next)

        self.experience_replay.add((obs, action, reward, obs_next, done))
        obs = obs_next

        b_obs, _, b_reward, b_obs_next, b_done = self.experience_replay.sample(self.batch_size)

        # TODO: problem here, the loss is always super small
        # DDQN loss calculation
        qmain_next = self.main.predict_on_batch(b_obs_next)
        qtarget_next = self.target.predict_on_batch(b_obs_next)
        a_max = np.argmax(qmain_next, axis=1)

        # print('main', qmain_next)
        # print('target', qtarget_next)

        y = b_reward + (self.gamma * ~b_done * qtarget_next[range(self.batch_size), a_max])
        y_sparse = np.zeros_like(qtarget_next)
        y_sparse[range(self.batch_size), a_max] = y

        # qtarget_next[range(self.batch_size), a_max] = y
        y_sparse[range(self.batch_size), a_max] = y

        loss = self.main.train_on_batch(b_obs, y_sparse)

        # Updates
        self.steps_taken += 1
        if self.steps_taken % self.target_update_freq == 0:
            print("Updating target weights, self.steps_taken", self.steps_taken)
            self.target.set_weights(self.main.get_weights())

        return obs, done, dict(loss=loss, action=action, reward=reward)
