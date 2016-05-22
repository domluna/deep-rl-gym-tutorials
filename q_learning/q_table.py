"""
Deep Q-Learning

https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

Hyperparams from original Deepmind Atari Paper
----------------------------------------------

minibatch size = 32
replay memory size = 1,000,000
agent history length = 4
target network update frequency = 10,000
discount factor = 0.99
action repeat = 4
update frequency = 4
learning rate = 0.00025 
gradient momentum = 0.95
squared gradient momentum = 0.95
min squared gradient = 0.01
initial exploration = 1
final exploration = 0.1
final exploration frame = 1,000,000
replay start size = 50,000
no-op max = 30

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
from keras import backend as K
from keras.layers.convolutional import Convolution2D
from keras.layers import Input, Dense
from keras.models import Model

import gym
import numpy as np
import tensorflow as tf
import logging
import argparse

class DeepmindDQN(object):
    """
    input 4 x 84 x 84 (image is 84x84)
    32 filters 8 x 8 stride 4
    64 filters 4 x 4 stride 2
    64 filters 3 x 3 stride 1
    fc 512 units
    fc 4-18 units (actions)

    """
    def __init__(self):
        pass

    def act(self):
        pass

    def fit(self):
        pass
