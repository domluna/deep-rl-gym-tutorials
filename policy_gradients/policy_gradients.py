"""
Policy Gradients

1. Sample paths.
2. Process paths (compute advantage, baseline, rewards, etc)
3. Run the paths through the policy (function approximator)
4. Compute gradients/update policy model weights
5. Profit?!?!

How we optimize the policy
--------------------------

L(theta) = sum t=0 to T-1 log policy(action_t | state_t, theta) * A_t
R_t = (sum u=t to T reward_u)
B_t = E [ sum u=t to T lambda^(u-t) * reward_u | state_t]
A_t = R_t - B_t

R_t = reward
A_t = advantage
B_t = baseline
theta = parameters of our policy, most like neural network weights.

The baseline can be thought of as the value function (V). When we evaluate the baseline
of a state we're predict how good our future returns will be given our current state.
So, intuitively if A_t > 0 that means the path we sampled is better than the expectation of
paths from the current state. Likewise, if A_t < 0, it's worse. Concretely, if A_t > 0 we want
more paths like that, if A_t < 0 we want less paths like that. Theta will be updated during training
to reflect this.


Types of parameterized policies
-------------------------------

Map s (state) to an output vector u

1. If the action is from a discrete set, the network maps s to a vector of probabilities (softmax)
2. If the action is continuous, then we map s to the mean/variance of a Gaussian distribution
(diagonal covariance that does not depend on s)
3. If a is binary valued, we use a single output, the probability of outputting 1 (although
we could also just use 1.)

TODO: implement baseline
TODO: implement generalized advantage estimation
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range

import gym
from gym.spaces import Box, Discrete
import tensorflow as tf
import numpy as np
from scipy.signal import lfilter
import argparse

def flatten_space(space):
    if isinstance(space, Box):
        return np.prod(space.shape)
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise ValueError("Env must be either Box or Discrete.")

def discount_cumsum(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def MLP(in_dim, out_dim, hidden_size, initializer,
        hidden_nonlin=tf.tanh,
        out_nonlin=tf.nn.softmax):
    """Simple MultiLayer Perceptron"""
    def _f(x):
        W1 = tf.Variable(initializer([in_dim, hidden_size]))
        W2 = tf.Variable(initializer([hidden_size, out_dim]))
        b1 = tf.Variable(tf.zeros([hidden_size]))
        b2 = tf.Variable(tf.zeros([out_dim]))
        h1 = hidden_nonlin(tf.nn.xw_plus_b(x, W1, b1))
        return out_nonlin(tf.nn.xw_plus_b(h1, W2, b2))
    return _f


class CategoricalPolicy(object):
    def __init__(self, mlp, optimizer, session):

        # Placeholder Inputs
        self._observations = tf.placeholder(tf.float32, name="observations")
        self._actions = tf.placeholder(tf.int32, name="actions")
        self._advantages = tf.placeholder(tf.float32, name="advantages")

        self._opt = optimizer
        self._sess = session

        probs = mlp(self._observations)

        # NOTE: Doesn't currently work due to gather_nd gradient not being currently implemented
        # inds = tf.transpose(tf.pack([tf.range(tf.shape(probs)[0]), self._actions]))
        # log_lik = tf.log(tf.gather_nd(probs, inds))

        idxs_flattened = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1] + self._actions
        log_lik = tf.log(tf.gather(tf.reshape(probs, [-1]), idxs_flattened) + 1e-8)

        act_op = tf.argmax(probs, 1, name="act_op")
        loss_op = tf.reduce_mean(log_lik * self._advantages, name="loss_op")
        train_op = self._opt.minimize(loss_op, name="train_op")

        self._act_op = act_op
        self._loss_op = loss_op
        self._train_op = train_op

    def act(self, observation):
        # expect observation to be shape(1, self.observation_space)
        a = self._sess.run(self._act_op, feed_dict={self._observations: observation})
        return a[0] 

    def train(self, observations, actions, advantages):
        loss, _ = self._sess.run([self._loss_op, self._train_op], feed_dict={self._observations:observations, self._actions:actions, self._advantages:advantages})
        return loss


class PolicyOptimizer(object):
    def __init__(self, env, policy, baseline, n_iter, n_episode, path_length,
        gamma=.99):

        self.policy = policy
        self.baseline = baseline
        self.env = env
        self.n_iter = n_iter
        self.n_episode = n_episode
        self.path_length = path_length
        self.gamma = gamma

    def sample_path(self):
        obs = []
        actions = []
        rewards = []
        ob = self.env.reset()

        for _ in range(self.path_length):
            a = self.policy.act(ob.reshape(1, -1))
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

        obs = np.concatenate([ p["observations"] for p in paths ])
        actions = np.concatenate([ p["actions"] for p in paths ])
        rewards = np.concatenate([ p["rewards"] for p in paths ])
        advantages = np.concatenate([ p["advantages"] for p in paths ])

        # TODO: fit baseline
        # self.baseline.fit(paths)

        return dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
        )


    def train(self):
        for i in range(1, self.n_iter+1):
            paths = []
            for _ in range(self.n_episode):
                paths.append(self.sample_path())
            data = self.process_paths(paths)
            loss = self.policy.train(data["observations"], data["actions"], data["advantages"])
            avg_return = np.mean([sum(p["rewards"]) for p in paths])
            print("Iteration {}: Loss = {}, Average Return = {}".format(i, loss, avg_return))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', default=100, type=int, help='number of iterations')
    parser.add_argument('--n_episode', default=100, type=int, help='number of episodes/iteration')
    parser.add_argument('--learning_rate', default=1e-1, help='learning rate for Adam Optimizer')
    parser.add_argument('--algorithm', default='Vanilla Policy Gradient', help='algorithm identifier')
    parser.add_argument('--outdir', default='vanilla-policy-gradient-CartPole-v0', type=str, help='output directory where results are saved (/tmp/ prefixed)')
    parser.add_argument('--upload', action='store_true', help='upload results via OpenAI Gym API')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    env = gym.make('CartPole-v0')
    outdir = '/tmp/' + args.outdir
    env.monitor.start(outdir, force=True)

    sess = tf.Session()

    horizon = env.spec.timestep_limit
    in_dim = flatten_space(env.observation_space)
    out_dim = flatten_space(env.action_space)
    hidden_size = 8

    initializer = tf.contrib.layers.xavier_initializer()
    mlp = MLP(in_dim, out_dim, hidden_size, initializer)

    opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    policy = CategoricalPolicy(mlp, opt, sess)
    po = PolicyOptimizer(env, policy, 0, args.n_iter, args.n_episode, horizon)

    sess.run(tf.initialize_all_variables())

    # train the policy optimizer
    po.train()

    env.monitor.close()

    # make sure to setup your OPENAI_GYM_API_KEY environment variable
    if args.upload:
        gym.upload(outdir, algorithm_id=args.algorithm)
