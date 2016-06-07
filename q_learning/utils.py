from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import warnings

from six.moves import range
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import img_as_ubyte

def preprocess(observation, new_height, new_width):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return img_as_ubyte(resize(rgb2gray(observation), (new_height, new_width)))

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
    save_path = saver.save(sess, dir + '/graph', step)
    print("Models saved in file: {} ...".format(save_path))

def noop_start(env, replay, buf, max_actions=30):
    """
    SHOULD BE RUN AT THE START OF AN EPISODE
    """
    obs = env.reset()
    for _ in range(np.random.randint(replay.history_window, max_actions)):
        next_obs, reward, terminal, _ = env.step(0) # 0 is a noop action in Atari envs
        replay.add((obs, 0, reward, terminal))
        buf.add(obs)
        obs = next_obs
    return obs

def random_start(env, replay, n):
    """Sample and add `n` random actions to the Experience Replay.

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
