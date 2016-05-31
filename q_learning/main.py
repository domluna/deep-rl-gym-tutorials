""" Double Deep Q-Learning on Gym Atari Environments
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
from keras import backend as K
from keras.optimizers import Adam

from agents import DDQN
from experience_replay import ExperienceReplay
from models import atari_cnn
from environments import Env

import gym
import numpy as np
import tensorflow as tf
import argparse
import random

# observation pre-processing
def rgb2y_resize(input_shape, new_height, new_width, session):
    img = tf.placeholder(tf.float32, shape=input_shape)
    reshaped = tf.reshape(img, [1] + list(input_shape))
    rgb2y = tf.image.rgb_to_grayscale(reshaped)
    bilinear = tf.image.resize_bilinear(rgb2y, [new_height, new_width])
    return lambda x: session.run(bilinear, feed_dict={img: x})

# MSE clipped into [1.0, 1.0] range
def clipped_mse(y_true, y_pred):
    err = K.mean(K.square(y_pred - y_true), axis=-1)
    return K.clip(err, -1.0, 1.0)

gym_env = gym.make('Breakout-v0')
# outdir = '/tmp/DDQN-Atari'
# gym_env.monitor.start(outdir, force=True)

seed = 0
replay_capacity = 100000
replay_start = 10000
episodes = 50
input_shape = (210, 160, 3)
resized_shape = (84, 84, 1)
n_actions = gym_env.action_space.n
learning_rate = 1e-3
batch_size = 32
max_path_length = 10000
gamma = .99
epsilon = 1

np.random.seed(0)
tf.set_random_seed(0)
gym_env.seed(0)

# saver = tf.train.Saver()

with tf.Graph().as_default():
    sess = tf.Session()
    K.set_session(sess)

    obs_preprocess = rgb2y_resize(input_shape, 84, 84, sess)
    reward_clip = lambda x: np.clip(x, -1.0, 1.0)

    env = Env(gym_env, obs_preprocess, reward_clip)

    main = atari_cnn(resized_shape, n_actions)
    adam = Adam(lr=learning_rate)
    main.compile(optimizer=adam, loss=clipped_mse)

    target = atari_cnn(resized_shape, n_actions)
    target.set_weights(main.get_weights())

    replay = ExperienceReplay(replay_capacity, batch_size, resized_shape)

    # and initialize replay with some random experiences
    print("Initializing replay with {} experiences".format(replay_start))

    obs = env.reset()
    for _ in range(replay_start):
        action = np.random.randint(n_actions)
        next_obs, reward, terminal, _ = env.step(action)
        replay.add((obs, action, reward, next_obs, terminal))
        if terminal:
            obs = env.reset()
        else:
            obs = next_obs

    print("Starting DDQN agent training")

    ql = DDQN(main, target, env, replay, batch_size,
            max_path_length=max_path_length,
            epsilon=epsilon) 

    for i in range(1, episodes+1):

        # no-op actions to start, up to 30
        obs = env.reset()
        for _ in range(np.random.randint(4, 30)):
            next_obs, reward, terminal, _ = env.step(0)
            replay.add((obs, action, reward, next_obs, terminal))
            obs = next_obs

        episode_reward = ql.run_episode()

        print("************************")
        print("Episode", i,)
        print("Episode total reward:", episode_reward)
        print("************************")

        if i % 10 == 0:
            print("Evaluating Agent ...")
            ql.play()

    sess.close()
    # env.monitor.close()






