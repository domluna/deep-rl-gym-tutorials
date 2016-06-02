from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range

import numpy as np

class DDQN:
    """
    Implements Double Deep Q-Learning

    a_max = arg max a' Q(s', a'; theta)
    y = r + gamma * Q(s', a_max, theta-)
    L = (y - Q(s, a; theta)) ** 2
    """
    def __init__(self, main, target, batch_size, n_actions, gamma=0.99):

        self.main = main
        self.target = target
        self.gamma = gamma
        self.n_actions = n_actions

    def predict_action(self, obs, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.choice(self.n_actions)
        else:
            qvals = self.main.predict(obs, batch_size=1)
            action = np.argmax(qvals)
        return action

    def train_on_batch(self, batch):
        b_obs, b_action, b_reward, b_next_obs, b_terminal = batch
        batch_size = b_obs.shape[0] # get batch shape

        next_main_qvals = self.main.predict_on_batch(b_next_obs)
        next_target_qvals = self.target.predict_on_batch(b_next_obs)
        a_max = np.argmax(next_main_qvals, axis=1)
        y = self.main.predict_on_batch(b_obs)
        y[range(batch_size), b_action] = b_reward + ~b_terminal * (self.gamma * next_target_qvals[range(batch_size), a_max])
        self.main.train_on_batch(b_obs, y)

    def update_target_weights(self):
        self.target.set_weights(self.main.get_weights())
