"""The main idea of CE (Cross Entropy) is to maintain a distribution
of possible solution, and update this distribution accordingly.

Preliminary investigation showed that applicability of CE to RL problems
is restricted severly by the phenomenon that the distribution concentrates to
a single point too fast.

To prevent this issue, noise is added to the previous stddev/variance update
calculation.

We implement two algorithms cem, the Cross-Entropy Method (CEM) with noise [1] and
Proportional Cross-Entropy (PCEM) [2].

CEM is implemented with decreasing variance noise

    variance + max(5 - t / 10, 0), where t is the iteration step

PCEM is implemented the same as CEM except we adjust the weights, evaluations of f
as follows:

    M = max(weights)
    m = min(weights)

    weights = (weight - m) / (M - m + eps)

    where eps is a very small value to avoid division by 0

An issue with CEM is it might not optimize the actual objective. PCEM helps
with this.

References:

    [1] Learning Tetris with the Noisy Cross-Entropy Method (Szita, Lorincz 2006)
    [2] The Cross-Entropy Method Optimizes for Quantiles (Goschin, Weinstein, Littman 2013)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range

import gym
import numpy as np
import logging
import argparse

# only two possible actions 0 or 1
class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]
    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a

def do_rollout(agent, env, num_steps, render=False):
    """
    Performs actions for num_steps on the environment
    based on the agents current params
    """
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1

# mean and std are 1D array of size d
def cem(f, mean, var, n_iters, n_samples, top_frac):
    top_n = int(np.round(top_frac * n_samples))
    for i in range(n_iters):
        # generate n_samples each iteration with new mean and stddev
        samples = np.transpose(np.array([np.random.normal(u, np.sqrt(o), n_samples) for u, o in zip(mean, var)]))
        ys = np.array([f(s) for s in samples])
        # the top samples are the ones which give the lowest f evaluation results
        top_idxs = ys.argsort()[::-1][:top_n]
        top_samples = samples[top_idxs]
        # this is taken straight from [1], constant noise param
        # dependent on the iteration step.
        v = max(5 - i / 10, 0)
        mean = top_samples.mean(axis=0)
        var = top_samples.var(axis=0) + v
        yield {'ys': ys, 'theta_mean': mean, 'y_mean': ys.mean()}

def pcem(f, mean, var, n_iters, n_samples, top_frac):
    eps = 1e-10 # avoid dividing by 0
    top_n = int(np.round(top_frac * n_samples))
    for i in range(n_iters):
        # generate n_samples each iteration with new mean and stddev
        samples = np.transpose(np.array([np.random.normal(u, np.sqrt(o), n_samples) for u, o in zip(mean, var)]))
        ys = np.array([f(s) for s in samples])
        max_y = np.max(ys)
        min_y = np.min(ys)
        ys = (ys - min_y) / (max_y - min_y + eps)
        # the top samples are the ones which give the lowest f evaluation results
        top_idxs = ys.argsort()[::-1][:top_n]
        top_samples = samples[top_idxs]
        # this is taken straight from [1], constant noise param
        # dependent on the iteration step.
        v = max(5 - i / 10, 0)
        mean = top_samples.mean(axis=0)
        var = top_samples.var(axis=0) + v
        yield {'ys': ys, 'theta_mean': mean, 'y_mean': ys.mean()}


def evaluation_func(policy, env, num_steps):
    def f(theta):
        agent = policy(theta)
        rew, t = do_rollout(agent, env, num_steps, render=False)
        return rew
    return f


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--iters', default=50, type=int, help='number of iterations')
    parser.add_argument('--samples', default=30, type=int, help='number of samples CEM algorithm chooses from on each iter')
    parser.add_argument('--num_steps', default=200, type=int, help='number of steps/actions in the rollout')
    parser.add_argument('--top_frac', default=0.2, type=float, help='percentage of top samples used to calculate mean and variance of next iteration')
    parser.add_argument('--algorithm', default='cem', type=str, choices=['pcem', 'cem'])
    parser.add_argument('--outdir', default='CartPole-v0-cem', type=str, help='output directory where results are saved (/tmp/ prefixed)')
    parser.add_argument('--render', action='store_true', help='show rendered results during training')
    parser.add_argument('--upload', action='store_true', help='upload results via OpenAI API')
    args = parser.parse_args()

    print(args)

    np.random.seed(args.seed)
    env = gym.make('CartPole-v0')
    num_steps = args.num_steps

    ef = None
    if args.algorithm == 'cem':
        ef = cem
    else:
        ef = pcem

    outdir = '/tmp/' + args.outdir
    env.monitor.start(outdir, force=True)

    f = evaluation_func(BinaryActionLinearPolicy, env, num_steps)

    # params for cem
    params = dict(n_iters=args.iters, n_samples=args.samples, top_frac=args.top_frac)
    u = np.random.randn(env.observation_space.shape[0]+1)
    var = np.square(np.ones_like(u) * 0.1)
    for (i, data) in enumerate(ef(f, u, var, **params)):
        print("Iteration {}. Episode mean reward: {}".format(i, data['y_mean']))
        agent = BinaryActionLinearPolicy(data['theta_mean'])
        if args.render:
            do_rollout(agent, env, num_steps, render=True)

    env.monitor.close()
    # make sure to setup your OPENAI_GYM_API_KEY environment variable
    if args.upload:
        gym.upload(outdir, algorithm_id=args.algorithm)
