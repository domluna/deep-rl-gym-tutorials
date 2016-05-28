from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import deque
from six.moves import range

import numpy as np
import random

class SimpleExperienceReplay:

    def __init__(self, capacity):
        self.replay = deque([], capacity)

    def add(self, experience):
        self.replay.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.replay, batch_size)
        obs = np.concatenate([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        obs_next = np.concatenate([e[3] for e in experiences])
        done = np.array([e[4] for e in experiences])

        return obs, actions, rewards, obs_next, done

    @property
    def capacity(self):
        return self.replay.maxlen

# TODO: implement this
class PrioritizedExperienceReplay:

    def __init__(self):
        raise NotImplementedError

    def add(self, experience):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

