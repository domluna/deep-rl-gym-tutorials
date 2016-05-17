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

# Parameterized Policies
# map s (state) to an output vector u, a is the action
#
# 1. If a is from a discrete set, the network maps s to a vector of probabilities
# (most likely a softmax activation)
# 2. If a is continuous, then we map s to the mean/variance of a Gaussian distribution
# (diagonal covariance that does not depend on s)
# 3. If a is binary valued, we use a single output, the probability of outputting 1
def CategoricalActionPolicy(object):
    def __init__(self, env, mlp):
        self._observations = tf.placeholder(tf.float32)
        self._actions = tf.placeholder(tf.float32)
        self._advantages = tf.placeholder(tf.float32)
        self.observation_space = self.env.observation_space.shape
        self.action_space = self.env.action_space.shape

        # simple 1 layer neural network
        self._W = tf.Variable(tf.random_normal((self.observation_space, self.action_space)))
        self._b = tf.Variable(tf.ones(self.action_space))

        y = tf.nn.xw_plus_b(observation, self._W, self._b)
        probs = tf.softmax(y)
        # TODO: figure out how to use gather_nd or
        tf.log()

        self._loss_op = tf.reduce_mean( * actions)
        self._act_op = tf.argmax(probs, 1)

    def act(self, observation):
        pass

    def loss(self, observations, actions, advantages):
        pass



# TODO: implemented model snapshotting
# so how it works is we sample paths
# process the paths to compute baselines, advantages, etc.
# put the processed paths through the policy, backprop, profit
# 1. sample paths
# 2. process paths (compute advantage, baseline, rewards, etc)
# 3. run the paths through the policy (function approximator)
# 4. ??
# 5. profit
#
class PolicyOptimizer(object):
    def __init__(self, policy, baseline, env, iters, rollouts, path_length,
        logger=None,
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
            observations=np.array(obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
        )

    def process_paths(self, paths):
        for p in paths:
            # TODO: compute baseline
            # b = self.baseline.predict(p)
            b = 0
            r = discount_cumsum(p["rewards"], self.gamma)
            a = r - b

            p["returns"] = r
            p["advantages"] = (a - a.mean()) / (a.std() + 1e-8) # normalize
            p["baselines"] = b

        # (rollouts, path_length)
        obs = np.concatenate([ p["observations"] for p in paths ])
        actions = np.concatenate([ p["actions"] for p in paths ])
        rewards = np.concatenate([ p["rewards"] for p in paths ])
        advantages = np.concatenate([ p["advantages"] for p in paths ])
        returns = np.concatenate([ p["returns"] for p in paths ])
        baselines = np.concatenate([ p["rewards"] for p in paths ])

        # TODO: fit baseline
        # self.baseline.fit(paths)

        return dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
        )


    # TODO: train here
    # self.policy is an MLP, the variables we train on are part of that
    # L(theta) = sum t=0 to T-1 log policy(action_t | state_t, theta) * A_t
    # A_t = (sum u=t to T reward_u) - b_t(state_t)
    # b_t = E [ sum u=t to T lambda^(u-t) * reward_u | state_t]
    # we can also think of the baseline as the value function V


    # The policy is an the neural network or whatever, so given the state and current
    # network params we produce an action. The api can be policy.act(observation). By default
    # the observation is the representation of the environment but we can obviously add to this.
    def train(self):
        paths = []
        for _ in range(self.rollouts):
            paths.append(self.sample_path())
        data = self.process_paths(paths)

        self.policy.train(observations, actions, advantages)

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', default=100, type=int, help='number of trajectories/iter to consider')
    parser.add_argument('--num_steps', default=100, type=int, help='length of a trajectory')
    parser.add_argument('--outdir', default='policy-gradient-CartPole-v0', type=str, help='output directory where results are saved (/tmp/ prefixed)')
    parser.add_argument('--render', action='store_true', help='show rendered results during training')
    parser.add_argument('--upload', action='store_true', help='upload results via OpenAI Gym API')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = gym.make('CartPole-v0')

    outdir = '/tmp/' + args.outdir
    env.monitor.start(outdir, force=True)

    # train here
    p = BinaryActionPolicy(env)
    opt = PolicyOptimizer()

    env.monitor.close()
    # make sure to setup your OPENAI_GYM_API_KEY environment variable
    if args.upload:
        gym.upload(outdir, algorithm_id=args.algorithm)
