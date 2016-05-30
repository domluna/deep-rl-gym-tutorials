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

from models import atari_cnn
from six.moves import range
from q_learner import DDQN
from experience_replay import SimpleExperienceReplay
from keras import backend as K
from keras.optimizers import Adam

import gym
import numpy as np
import tensorflow as tf
import argparse
import random

def process_img(new_height, new_width, session):
    def _f(img):
        img = tf.reshape(img, [1] + list(img.shape))
        img = tf.convert_to_tensor(img)
        rgb2y = tf.image.rgb_to_grayscale(img)
        bilinear = tf.image.resize_bilinear(rgb2y, [new_height, new_width])
        return bilinear
    return lambda img: session.run(_f(img))

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

env = gym.make('Pong-v0')
outdir = '/tmp/dump'
env.monitor.start(outdir, force=True)

replay_capacity = 100000
episodes = 100
input_shape = (84, 84, 1)
# input_shape = (210, 160, 3)
learning_rate = 1e-3
batch_size = 32
max_path_length = 200
n_actions = env.action_space.n
gamma = .99
# epsilon = tf.Variable(1.0, trainable=False, name='epsilon')
epsilon = 0.1
render = True

with tf.Graph().as_default():
    sess = tf.Session()
    K.set_session(sess)

    main = atari_cnn(input_shape, n_actions)
    adam = Adam(lr=learning_rate)
    main.compile(optimizer=adam, loss='mse')

    target = atari_cnn(input_shape, n_actions)
    target.set_weights(main.get_weights())

    obs_preprocess = process_img(84, 84, sess)
    # obs_preprocess = lambda x : x.reshape((1,) + x.shape)

    # setup experience replay and initialize with some
    # random experiences
    er = SimpleExperienceReplay(replay_capacity)

    obs = env.reset()
    obs = obs_preprocess(obs)
    for _ in range(batch_size):
        action = np.random.choice(n_actions)
        next_obs, reward, done, _ = env.step(action)
        next_obs = obs_preprocess(next_obs)
        er.add((obs, action, reward, next_obs, done))
        obs = next_obs

    ql = DDQN(main, target, env, er, obs_preprocess, 
            batch_size=batch_size,
            max_path_length=max_path_length,
            epsilon=epsilon) 

    for i in range(episodes):
        loss, episode_reward = ql.run_episode(render)
        lm = np.mean(loss)
        ls = np.std(loss)

        print("************************")
        print("Episode #:", i, "length =", len(loss))
        print("Episode loss mean:", lm)
        print("Episode loss stddev:", ls)
        print("Episode total reward:", episode_reward)
        print("************************")

    sess.close()
    env.monitor.close()






