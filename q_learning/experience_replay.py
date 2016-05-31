from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class ExperienceReplay(object):

    def __init__(self, capacity, batch_size, observation_shape):
        self.capacity = capacity
        self.batch_size = batch_size
        self.index = 0
        self.size = 0

        obs_memory_shape = [capacity] + list(observation_shape)
        obs_buffer_shape = [batch_size] + list(observation_shape)

        # memory
        self.obs = np.zeros(obs_memory_shape, dtype=np.float32)
        self.next_obs = np.zeros(obs_memory_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.bool)

        # buffer
        self.b_obs = np.zeros(obs_buffer_shape, dtype=np.float32)
        self.b_next_obs = np.zeros(obs_buffer_shape, dtype=np.float32)
        self.b_actions = np.zeros(batch_size, dtype=np.uint8)
        self.b_rewards = np.zeros(batch_size, dtype=np.float32)
        self.b_terminals = np.zeros(batch_size, dtype=np.bool)


    def add(self, experience):
        obs, action, reward, next_obs, terminal = experience

        self.obs[self.index, ...] = obs
        self.next_obs[self.index, ...] = next_obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = terminal

        self.index = (self.index + 1) % self.capacity
        if self.size < self.capacity: self.size += 1

    def sample(self):
        ints = np.random.choice(self.size, self.batch_size, False)

        self.b_obs[...] = self.obs[ints, ...]
        self.b_obs_next[...] = self.obs_next[ints, ...]
        self.b_actions[...] = self.actions[ints]
        self.b_rewards[...] = self.reward[ints]
        self.b_terminals[...] = self.terminals[ints]

        return self.b_obs, self.b_actions, self.b_rewards, self.b_next_obs, self.b_terminals




