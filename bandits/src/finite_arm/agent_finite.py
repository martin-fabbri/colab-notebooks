"""
Finite bandit agents
"""

import numpy as np

from base.agent import Agent
from base.agent import random_argmax

_SMALL_NUMBER = 1e-10


class FiniteBernoulliBanditEpsilonGreedy(Agent):
    """
    Simple agent made for finite armed bandit problems.
    """

    def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
        super().__init__()
        self.n_arm = n_arm
        self.epsilon = epsilon
        self.prior_success = np.array([a0 for arm in range(n_arm)])
        self.prior_failure = np.array([b0 for arm in range(n_arm)])

    def set_prior(self, prior_sucess, prior_failure):
        # overwrite the default prior
        self.prior_success = np.array(prior_sucess)
        self.prior_failure = np.array(prior_failure)

    def get_posterior_mean(self):
        return self.prior_success / (self.prior_success + self.prior_failure)

    def get_posterior_sample(self):
        return np.random.beta(self.prior_success, self.prior_failure)

    def update_observation(self, observation, action, reward):
        # naive error checking for compatibility with environment
        assert observation == self.n_arm

        if np.isclose(reward, 1):
            self.prior_success[action] += 1

        elif np.isclose(reward, 0):
            self.prior_failure[action] += 1
        else:
            raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

    def pick_action(self, observation):
        """
        Take random action prob epsilon, else be greedy
        :param observation:
        :return:
        """
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_arm)
        else:
            posterior_means = self.get_posterior_mean()
            action = random_argmax(posterior_means)

        return action


class FiniteBernoulliBanditTS(FiniteBernoulliBanditEpsilonGreedy):
    """
    Thompson sampling on finite armed bandit.
    """
    def pick_action(self, observation):
        """Thompson sampling with BEta posterior for action selection"""
        samples_means = self.get_posterior_sample()
        action = random_argmax(samples_means)
        return action
