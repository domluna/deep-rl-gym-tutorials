""" Double Deep Q-Learning on Gym Atari Environments
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
from keras import backend as K
from keras.optimizers import Adam

from agents import DDQN
from memory import SimpleExperienceReplay, Buffer
from models import duel_atari_cnn as nn
from environments import Env
from utils import *

import gym
import numpy as np
import tensorflow as tf
import argparse
import random
import time
import os

def clipped_mse(y_true, y_pred):
    """MSE clipped into [1.0, 1.0] range"""
    err = K.mean(K.square(y_pred - y_true), axis=-1)
    return K.clip(err, -1.0, 1.0)

def play(ql, env, buf, epsilon=0.0):
    terminal = False
    episode_reward = 0
    t = 0
    buf.reset()
    obs = env.reset()
    buf.add(obs)

    while not terminal:
        env.render()
        action = ql.predict_action(buf.state, epsilon)
        obs, reward, terminal, _ = env.step(action)
        buf.add(obs)
        if reward != 0:
            episode_reward += reward
        t += 1

    print("Episode Reward {}".format(episode_reward))

parser = argparse.ArgumentParser()
parser.add_argument('--games', type=int, default=10, help='Number of games played')
parser.add_argument('--epsilon', type=float, default=0, help='Epsilon value, probability of a random action')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for Adam Optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='Number of states to train on each step')
parser.add_argument('--gamma', type=float, default=0.99, help='Gamma for Q-Learning steps')

parser.add_argument('--height', type=int, default=80, help='Observation height after resize')
parser.add_argument('--width', type=int, default=80, help='Observation width after resize')
parser.add_argument('--history_window', type=int, default=4, help='Number of observations forming a state')

parser.add_argument('--checkpoint_dir', type=str, help='Directory TF Graph will be saved to periodically')

parser.add_argument('--name', type=str, help='Name of OpenAI environment to run, ex. (Breakout-v0, Pong-v0)')
parser.add_argument('--seed', type=int, default=0, help='Random seed')

args = parser.parse_args()
print(args)

if not args.checkpoint_dir:
    parser.error('--checkpoint_dir must not be empty')

if not args.name:
    parser.error('--name must not be empty')

gym_env = gym.make(args.name)

np.random.seed(args.seed)
random.seed(args.seed)
tf.set_random_seed(args.seed)
gym_env.seed(args.seed)

network_input_shape = (args.history_window, args.height, args.width)
n_actions = gym_env.action_space.n
observation_shape = gym_env.observation_space.shape

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    main = nn(network_input_shape, n_actions)
    target = nn(network_input_shape, n_actions)
    adam = Adam(lr=args.learning_rate)
    main.compile(optimizer=adam, loss=clipped_mse)

    saver = tf.train.Saver()
    load_checkpoint(saver, args.checkpoint_dir, sess)

    buf = Buffer(args.history_window, (args.height, args.width))
    obs_preprocess = lambda i: preprocess(i, args.height, args.width)
    reward_clip = lambda r: np.clip(r, -1.0, 1.0)
    env = Env(gym_env, obs_preprocess, reward_clip)

    ql = DDQN(main, target, args.batch_size, n_actions, args.gamma)

    print("Playing {} games ...".format(args.games))
    for _ in range(args.games):
        play(ql, env, buf, epsilon=args.epsilon)
