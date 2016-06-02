""" Double Deep Q-Learning on Gym Atari Environments
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
from keras import backend as K
from keras.optimizers import Adam
from skimage.color import rgb2gray
from skimage.transform import resize

from agents import DDQN
from memory import SimpleExperienceReplay, Buffer
from models import atari_cnn
from environments import Env

import gym
import numpy as np
import tensorflow as tf
import argparse
import random

seed = 0
total_steps = 1000000
replay_capacity = 100000
replay_start = 10000
target_update_frec = 10000
save_model_freq = 50000
history_window = 4
height = 84
width = 84
network_input_shape = (history_window, height, width)
learning_rate = 1e-4
batch_size = 32
gamma = .99

def preprocess(observation, new_height, new_width):
    return resize(rgb2gray(observation), (new_height, new_width))

def clipped_mse(y_true, y_pred):
    """MSE clipped into [1.0, 1.0] range"""
    err = K.mean(K.square(y_pred - y_true), axis=-1)
    return K.clip(err, -1.0, 1.0)

def play(ql, env, buffer, epsilon=0.05, render=True):
    terminal = False
    episode_reward = 0
    t = 0
    buffer.reset()
    obs = env.reset()
    buffer.add(obs)

    while not terminal:
        if render: env.render()
        action = ql.predict_action(buffer.observations, epsilon)
        obs, reward, terminal, _ = env.step(action)
        buffer.add(obs)
        episode_reward += reward
        t += 1

    print("Episode Reward {}".format(episode_reward))

def noop_start(env, replay, buffer, max_actions=30):
    """
    SHOULD BE RUN AT THE START OF AN EPISODE
    """
    obs = env.reset()
    for _ in range(np.random.randint(replay.history_window, max_actions)):
        next_obs, reward, terminal, _ = env.step(0)
        replay.add((obs, 0, reward, terminal))
        buffer.add(obs)
        obs = next_obs
    return obs

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
save_path = '/tmp/atari.ckpt'

# seed all the things! DJ Khaled says this is a "major key"
np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)
gym_env.seed(seed)

with tf.Graph().as_default():
    sess = tf.Session()
    K.set_session(sess)

    epsilon_decay = lambda t: max(0.1, 1.0 - (t/(total_steps+1)))
    main = atari_cnn(network_input_shape, n_actions)
    target = atari_cnn(network_input_shape, n_actions)
    adam = Adam(lr=learning_rate)
    main.compile(optimizer=adam, loss=clipped_mse)
    saver = tf.train.Saver()

    replay = SimpleExperienceReplay(replay_capacity, batch_size, history_window, (height, width))
    buffer = Buffer(history_window, (height, width))
    obs_preprocess = lambda i: preprocess(i, height, width)
    reward_clip = lambda r: np.clip(r, -1.0, 1.0)
    env = Env(gym_env, obs_preprocess, reward_clip)

    # and initialize replay with some random experiences
    print("Initializing replay with {} experiences".format(replay_start))
    random_start(env, replay, replay_start)

    print("Starting DDQN agent training")
    ql = DDQN(main, target, batch_size, n_actions, gamma=gamma)
    ql.update_target_weights()

    terminal = False
    episode_reward = 0
    episode_n = 1
    episode_len = 0

    # resets env and does NOOP (0) actions
    obs = noop_start(env, replay, buffer)
    for t in range(1, total_steps+1):
        if terminal:
            print("************************")
            print("Episode: {}".format(episode_n))
            print("Episode total reward = {}".format(episode_reward))
            print("Number of actions taken during episode = {}".format(episode_len))

            episode_n += 1
            episode_reward = 0
            episode_len = 0
            obs = noop_start(env, replay, buffer)

        epsilon = epsilon_decay(t)

        buffer.add(obs)
        action = ql.predict_action(buffer.observations, epsilon)
        obs_next, reward, terminal, _ = env.step(action)
        replay.add((obs, action, reward, terminal))
        episode_reward += reward
        episode_len += 1
        obs = obs_next

        batch = replay.sample()
        ql.train_on_batch(batch)

        if t % target_update_frec == 0:
            print("Updating Target Weights ...")
            ql.update_target_weights()

        if t % save_model_freq == 0:
            print("{} steps in ...".format(t))
            print("Models saved in file: {} ...".format(save_path))
            saver.save(sess, save_path)

    sess.close()
    # env.monitor.close()






