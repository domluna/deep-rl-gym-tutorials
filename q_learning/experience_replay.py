from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import deque
from six.moves import range

import numpy as np
import random

class ExperienceReplay(object):
    def add(self, experience):
        """
        Add an experience to the experience replay.

        An experience should be a tuple of the form:
            (observation, action, reward, next_observation, terminal)
        """
        raise NotImplementedError

    def sample(self, batch_size):
        """
        Sample `batch_size` experiences for the experience replay
        """
        raise NotImplementedError

class SimpleExperienceReplay(ExperienceReplay):

    def __init__(self, capacity):
        # self.replay = deque([], capacity)
        self.replay = []

    def add(self, experience):
        self.replay.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.replay, batch_size)
        obs = np.concatenate([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_obs = np.concatenate([e[3] for e in experiences])
        terminal = np.array([e[4] for e in experiences])

        return obs, actions, rewards, next_obs, terminal

    @property
    def capacity(self):
        return self.replay.maxlen

# TODO: implement this
class PrioritizedExperienceReplay(ExperienceReplay):

    def __init__(self):
        raise NotImplementedError

    def add(self, experience):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

