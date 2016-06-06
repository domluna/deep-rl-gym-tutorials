"""Double Deep Q-Learning on Gym Atari Environments
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
from envs import Env
from utils import *

import gym
import numpy as np
import tensorflow as tf
import argparse
import random
import time

def clipped_mse(y_true, y_pred):
    """MSE clipped into [1.0, 1.0] range"""
    err = K.mean(K.square(y_pred - y_true), axis=-1)
    return K.clip(err, -1.0, 1.0)

parser = argparse.ArgumentParser()
parser.add_argument('--total_steps', type=int, default=5000000, help='Number of total training steps')
parser.add_argument('--exploration_steps', type=int, default=1000000, help='Number of exploration steps with epsilon=1.0')
parser.add_argument('--epsilon', type=float, default=0.1, help='Fixed epsilon value after exploration phase')

parser.add_argument('--learning_rate', type=float, default=0.00025, help='Learning rate for Adam Optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='Number of states to train on each step')
parser.add_argument('--gamma', type=float, default=0.99, help='Gamma for Q-Learning steps')
parser.add_argument('--target_update_freq', type=int, default=10000, help='Frequency to update target network weights')
parser.add_argument('--train_batch_freq', type=int, default=4, help='Frequency to train a batch of states (steps)')

parser.add_argument('--replay_capacity', type=int, default=1000000, help='Maximum capacity of the experience replay')
parser.add_argument('--replay_start', type=int, default=50000, help='Number of random action observations to intially fill experience replay')
parser.add_argument('--height', type=int, default=80, help='Observation height after resize')
parser.add_argument('--width', type=int, default=80, help='Observation width after resize')
parser.add_argument('--history_window', type=int, default=4, help='Number of observations forming a state')

parser.add_argument('--save_model_freq', type=int, default=250000, help='Frequency to save TF Graph')
parser.add_argument('--checkpoint_dir', type=str, help='File TF Graph will be saved to periodically')
parser.add_argument('--monitor_dir', type=str, help='Directory OpenAI Gym will monitor and write results to')
parser.add_argument('--resume', action='store_true', help='Load saved model from checkpoint_dir and continue monitoring from monitor_dir')

parser.add_argument('--name', type=str, help='Name of OpenAI environment to run, ex. (Breakout-v0, Pong-v0)')
parser.add_argument('--seed', type=int, default=0, help='Random seed')

args = parser.parse_args()
print(args)

if not args.name:
    parser.error("--name must be not be empty")

gym_env = gym.make(args.name)

np.random.seed(args.seed)
random.seed(args.seed)
tf.set_random_seed(args.seed)
gym_env.seed(args.seed)

network_input_shape = (args.history_window, args.height, args.width)
n_actions = gym_env.action_space.n
observation_shape = gym_env.observation_space.shape
epsilon = 1.0

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    t = tf.Variable(0, trainable=False, name='step')
    incr_t = tf.assign_add(t, 1)

    main = nn(network_input_shape, n_actions)
    target = nn(network_input_shape, n_actions)
    adam = Adam(lr=args.learning_rate, epsilon=0.1)
    main.compile(optimizer=adam, loss=clipped_mse)

    # compile initializes the network vars
    sess.run(tf.initialize_variables([t]))

    # TODO: Figure out why after a lot of steps the meta
    # file becomes so large, for now set max_to_keep=1.
    saver = tf.train.Saver(max_to_keep=1)
    if args.resume and args.checkpoint_dir:
        load_checkpoint(saver, args.checkpoint_dir, sess)

    replay = SimpleExperienceReplay(args.replay_capacity, args.batch_size, args.history_window, (args.height, args.width))
    buf = Buffer(args.history_window, (args.height, args.width))
    obs_preprocess = lambda o: preprocess(o, args.height, args.width)
    reward_clipper = lambda r: np.clip(r, -1.0, 1.0)

    # wrap Gym Env so we can easily process observations
    # and rewards
    env = Env(gym_env, obs_preprocess, reward_clipper)

    # Initialize replay with some random experiences
    print("Initializing replay with {} experiences".format(args.replay_start))
    random_start(env, replay, args.replay_start)

    print("Training DDQN Agent")
    if args.monitor_dir:
        if args.resume:
            env.monitor.start(args.monitor_dir, resume=True)
        else:
            env.monitor.start(args.monitor_dir, force=True)

    ql = DDQN(main, target, args.batch_size, n_actions, args.gamma)
    ql.update_target_weights()

    terminal = False
    episode_reward = 0
    episode_n = 1
    episode_len = 0

    # Resets env and does NOOP (0) actions
    obs = noop_start(env, replay, buf)
    t0 = time.time()
    t_val = sess.run(t)
    while t_val < args.total_steps:
        if terminal:
            taken = time.time() - t0
            print("************************")
            print("Episode {}".format(episode_n))
            print("Total reward = {}".format(episode_reward))
            print("Number of actions = {}".format(episode_len))
            print("Time taken = {:.2f} seconds".format(taken))

            episode_n += 1
            episode_reward = 0
            episode_len = 0
            t0 = time.time()
            obs = noop_start(env, replay, buf)

        if t_val > args.exploration_steps:
            epsilon = args.epsilon

        buf.add(obs)
        action = ql.predict_action(buf.state, epsilon)
        obs_next, reward, terminal, _ = env.step(action)
        replay.add((obs, action, reward, terminal))
        if reward != 0:
            episode_reward += reward
        episode_len += 1
        obs = obs_next

        if t_val % args.train_batch_freq == 0:
            batch = replay.sample()
            ql.train_on_batch(batch)

        sess.run(incr_t)
        t_val = sess.run(t)

        if t_val % args.target_update_freq == 0:
            ql.update_target_weights()
            print("Updated Target Weights ...")

        if t_val % args.save_model_freq == 0 and args.checkpoint_dir:
            save_checkpoint(saver, args.checkpoint_dir, sess, t)

    sess.close()
