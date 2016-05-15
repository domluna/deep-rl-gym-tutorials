"""
Policy Gradients
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range

import gym
import tensorflow as tf
import numpy as np
from scipy.signal import lfilter
import logging
import argparse

def discount_cumsum(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def discount_sum(x, gamma):
    return np.sum(x * gamma ** np.arange(len(x)))

# TODO: implemented model snapshotting
# so how it works is we sample paths
# process the paths to compute baselines, advantages, etc.
# put the processed paths through the policy, backprop, profit
# 1. sample paths
# 2. process paths (compute advantage, baseline, rewards, etc)
# 3. run the paths through the policy (function approximator)
# 4. ??
# 5. profit
class PolicyOptimizer(object):
    def __init__(self, policy, baseline, env, iters, rollouts, path_length,
        gamma=.99,
        session=None):

        self.policy = policy
        self.baseline = baseline
        self.env = env
        self.iters = iters
        self.rollouts = rollouts
        self.path_length = path_length
        self.gamma = gamma
        self.sess = session

        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def sample_path(self):
        obs = []
        actions = []
        rewards = []
        ob = self.env.reset()

        for _ in range(self.path_length):
            a = self.policy.act(ob)
            (next_ob, r, done, _) = self.env.step(a)
            obs.append(ob)
            actions.append(a)
            rewards.append(r)
            ob = next_ob
            if done:
                break

        return dict(
            observations: np.array(obs),
            actions: np.array(actions),
            rewards: np.array(rewards),
        )

    # TODO: process
    def process_paths(self, paths):
        for p in paths:
            # TODO: compute baseline
            # if self.baseline:
            #     pass
            # else:
            #     pass
            b = 0 # TODO: baseline
            r = discount_cumsum(p["rewards"], self.gamma)
            a = r - b

            p["returns"] = r
            p["advantages"] = (a - a.mean()) / (a.std() + 1e-8) # normalize
            p["baselines"] = b

        obs = np.concatenate(p["observations"] for p in paths)
        actions = np.concatenate(p["actions"] for p in paths)
        rewards = np.concatenate(p["rewards"] for p in paths)
        advantages = np.concatenate(p["advantages"] for p in paths)
        returns = np.concatenate(p["returns"] for p in paths)
        baselines = np.concatenate(p["rewards"] for p in paths)


    # TODO: train here
    # self.policy is an MLP, the variables we train on are part of that
    # L(theta) = sum t=0 to T-1 log policy(action_t | state_t, theta) * A_t
    # A_t = (sum u=t to T reward_u) - b_t(state_t)
    # b_t = E [ sum u=t to T lambda^(u-t) * reward_u | state_t]


    # The policy is an the neural network or whatever, so given the state and current
    # network params we produce an action. The api can be policy.act(observation). By default
    # the observation is the representation of the environment but we can obviously add to this.
    def train(self):
        pass

def foo(iters, num_steps):


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', default=100, type=int, help='number of trajectories/iter to consider')
    parser.add_argument('--num_steps', default=100, type=int, help='length of a trajectory')
    parser.add_argument('--outdir', default='CartPole-v0-pg', type=str, help='output directory where results are saved (/tmp/ prefixed)')
    parser.add_argument('--render', action='store_true', help='show rendered results during training')
    parser.add_argument('--upload', action='store_true', help='upload results via OpenAI API')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()

    print(args)

    np.random.seed(args.seed)
    env = gym.make('CartPole-v0')

    outdir = '/tmp/' + args.outdir
    env.monitor.start(outdir, force=True)

    env.monitor.close()
    # make sure to setup your OPENAI_GYM_API_KEY environment variable
    if args.upload:
        gym.upload(outdir, algorithm_id=args.algorithm)
