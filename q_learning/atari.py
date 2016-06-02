""" Double Deep Q-Learning on Gym Atari Environments
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
from keras import backend as K
from keras.optimizers import Adam
from collections import Counter

from agents import DDQN
from memory import SimpleExperienceReplay
from models import atari_cnn
from environments import Env

import gym
import numpy as np
import tensorflow as tf
import argparse
import random

seed = 0
replay_capacity = 100000
replay_start = 1000
target_update_frec = 10000
episodes = 50
history_window = 4
height = 84
width = 84
network_input_shape = (history_window, height, width)
learning_rate = 1e-3
batch_size = 32

# DDQN specific
gamma = .99

# steps to train for
total_steps = 100000

# observation pre-processing
def rgb2y_resize(input_shape, new_height, new_width, session):
    img = tf.placeholder(tf.float32, shape=input_shape)
    reshaped = tf.reshape(img, [1] + list(input_shape))
    rgb2y = tf.image.rgb_to_grayscale(reshaped)
    bilinear = tf.image.resize_bilinear(rgb2y, [new_height, new_width])
    squeezed = tf.squeeze(bilinear)
    return lambda x: session.run(squeezed, feed_dict={img: x})

def clipped_mse(y_true, y_pred):
    """MSE clipped into [1.0, 1.0] range"""
    err = K.mean(K.square(y_pred - y_true), axis=-1)
    return K.clip(err, -1.0, 1.0)

def play(ql, env, epsilon=0.05, render=True):
    obs = self.env.reset()

    actions = Counter()
    terminal = False
    total_reward = 0
    t = 0

    while not terminal:
        action = ql.predict_action(obs, epsilon)
        obs, reward, terminal, _ = env.step(action)
        total_reward += reward
        actions[action] += 1
        if t % 3 == 0: self.env.render()
        t += 1

    print('Action statistics:', actions)

def noop_start(env, replay, max_actions=30):
    """
    SHOULD BE RUN AT THE START OF AN EPISODE
    """
    obs = env.reset()
    for _ in range(np.random.randint(replay.history_window, max_actions)):
        next_obs, reward, terminal, _ = env.step(0)
        replay.add((obs, action, reward, terminal))
        obs = next_obs

def random_start(env, replay, n):
    """Add `n` random actions to the Replay Experience.

    If a terminal state is reached, the environmment will reset
    and sampling with continue.
    """
    obs = env.reset()
    for _ in range(n):
        action = env.action_space.sample()
        next_obs, reward, terminal, _ = env.step(action)
        replay.add((obs, action, reward, terminal))
        if terminal: 
            obs = env.reset()
        else:
            obs = next_obs


gym_env = gym.make('Breakout-v0')
n_actions = gym_env.action_space.n
observation_shape = gym_env.observation_space.shape
# outdir = '/tmp/DDQN-Atari'
# gym_env.monitor.start(outdir, force=True)

# seed all the things! DJ Khaled says this is a "major key"
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)
gym_env.seed(0)

# saver = tf.train.Saver()

with tf.Graph().as_default():
    sess = tf.Session()
    K.set_session(sess)

    obs_preprocess = rgb2y_resize(observation_shape, height, width, sess)
    reward_clip = lambda x: np.clip(x, -1.0, 1.0)
    epsilon_decay = lambda t: max(0.1, 1.0 - (t/total_steps+1))
    env = Env(gym_env, obs_preprocess, reward_clip)

    main = atari_cnn(network_input_shape, n_actions)
    target = atari_cnn(network_input_shape, n_actions)
    adam = Adam(lr=learning_rate)
    main.compile(optimizer=adam, loss=clipped_mse)

    replay = SimpleExperienceReplay(replay_capacity, batch_size, history_window, height, width)

    # and initialize replay with some random experiences
    print("Initializing replay with {} experiences".format(replay_start))
    random_start(env, replay, replay_start)

    print("Starting DDQN agent training")
    ql = DDQN(main, target, batch_size, n_actions, gamma=gamma)
    ql.update_target_weights()

    terminal = False
    episode_reward = 0
    episode_n = 1

    obs = env.reset()
    for t in range(1, total_steps+1):

        env.render()

        epsilon = epsilon_decay(t)
        print(epsilon)

        action = ql.predict_action(obs, epsilon)
        obs_next, reward, terminal, _ = env.step(action)
        replay.add((obs, action, reward, terminal))

        batch = replay.sample()
        ql.train_on_batch(batch)

        if terminal:
            print("************************")
            print("Episode: {}".format(episode_n))
            print("Episode total reward = {}".format(episode_reward))
            print()

            episode_n += 1
            episode_reward = 0
            terminal = False

            # resets env and does NOOP (0) actions
            noop_start(env, replay)
        else:
            obs = obs_next
            episode_reward += reward

        if t % target_update_frec == 0:
            print("Evaluating Agent ...")
            ql.play()
            ql.update_target_weights()

        if t % 1000 == 0:
            print("Played {} steps ...".format(t))

    sess.close()
    # env.monitor.close()






