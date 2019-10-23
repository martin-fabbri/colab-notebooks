import numpy as np


class Environment(object):
    """Base class for all bandits environment."""

    def __init__(self):
        """Initiate the environment."""

    def get_observation(self):
        """Returns an observation from the environment."""
        pass

    def get_optimal_reward(self):
        """Gets the expected reward of an action."""

    def get_expected_reward(self, action):
        """Gets the expected reward of an action."""
        pass

    def get_stochastic_reward(self, action):
        """Gets a stochastic reward for the action"""
        pass

    def advance(self, action, reward):
        """Updating the environment (useful for nonstationary bandit)."""
        pass
