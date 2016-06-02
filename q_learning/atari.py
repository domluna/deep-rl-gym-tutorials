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
import time
import os

seed = 0
max_steps = 100000
replay_capacity = 100000
replay_start = 1000
target_update_frec = 10000
save_model_freq = 10000
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

def load_checkpoint(saver, dir, sess):
    ckpt = tf.train.get_checkpoint_state(dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restored from checkpoint {}".format(ckpt.model_checkpoint_path))
    else:
        print("No checkpoint")

def save_checkpoint(saver, dir, sess, step=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_path = saver.save(sess, checkpoint_dir + '/model', step)
    print("Models saved in file: {} ...".format(save_path))


env_name = 'Breakout-v0'
gym_env = gym.make(env_name)
outdir = '/tmp/DDQN-monitor' + '-' + env_name
gym_env.monitor.start(outdir, force=True)
n_actions = gym_env.action_space.n
observation_shape = gym_env.observation_space.shape
checkpoint_dir = '/tmp/DDQN-ckpt' + '-' +  env_name

np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)
gym_env.seed(seed)

with tf.Graph().as_default():
    sess = tf.Session()
    K.set_session(sess)

    t = tf.Variable(0, trainable=False, name='step')
    epsilon = tf.Variable(1.0, trainable=False, name='epsilon')
    epsilon_decay = tf.assign(epsilon, tf.maximum(0.1, 1.0 - tf.cast(t / max_steps, tf.float32)))
    incr_t = tf.assign_add(t, 1)

    main = atari_cnn(network_input_shape, n_actions)
    target = atari_cnn(network_input_shape, n_actions)
    adam = Adam(lr=learning_rate)
    main.compile(optimizer=adam, loss=clipped_mse)

    sess.run(tf.initialize_variables([t, epsilon]))

    saver = tf.train.Saver()
    load_checkpoint(saver, checkpoint_dir, sess)

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
    t0 = time.time()
    t_val = sess.run(t)
    while t_val < max_steps:
        if terminal:
            taken = time.time() - t0
            print("************************")
            print("Episode {}".format(episode_n))
            print("total reward = {}".format(episode_reward))
            print("Number of actions = {}".format(episode_len))
            print("Time taken = {:.2f} seconds".format(taken))
            t0 = time.time()

            episode_n += 1
            episode_reward = 0
            episode_len = 0
            obs = noop_start(env, replay, buffer)

        sess.run(epsilon_decay)
        eps_val = sess.run(epsilon)

        buffer.add(obs)
        action = ql.predict_action(buffer.observations, eps_val)
        obs_next, reward, terminal, _ = env.step(action)
        replay.add((obs, action, reward, terminal))
        episode_reward += reward
        episode_len += 1
        obs = obs_next

        batch = replay.sample()
        ql.train_on_batch(batch)

        sess.run(incr_t)
        t_val = sess.run(t)

        if t_val % target_update_frec == 0:
            ql.update_target_weights()
            print("Updated Target Weights ...")

        if t_val % save_model_freq == 0:
            save_checkpoint(saver, checkpoint_dir, sess, t)
            print("Epsilon status =  {:.3f}".format(eps_val))


    sess.close()
    env.monitor.close()






