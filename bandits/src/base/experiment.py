"""
Experiment wraps an agent with an agent with environment and then runs the experiment.

We end up with several experiment variants since we might want to log different elements
of the agent/environment interaction. At the end of 'run_experiment' we save the key
result for plotting in a pandas in a dataframe 'experiment.results'.
"""

import numpy as np
import pandas as pd


class BaseExperiment(object):
    """
    Simple experiment that logs regret and action taken.
    """

    def __init__(self, agent, environment, n_steps, seed=0, rec_freq=1, unique_id='NULL'):
        """
        Setting up the experiment.
        Note that unique_id should be used to identify the job later for analysis.
        :param agent:
        :param environment:
        :param n_steps:
        :param seed:
        :param rec_freq:
        :param unique_id:
        """
        self.agent = agent
        self.environment = environment
        self.n_steps = n_steps
        self.seed = seed
        self.unique_id = unique_id

        self.results = []
        self.data_dict = {}
        self.rec_freq = rec_freq

    def run_step_maybe_log(self, t):
        # evolve the bandit (potential contextual) for one step and pick action
        observation = self.environment.get_observation()
        action = self.agent.pick_action(observation)

        # compute useful stuff for regret calculations
        optimal_reward = self.environment.get_optimal_reward()
        expected_reward = self.environment.get_expected_reward(action)
        reward = self.environment.get_stochastic_reward(action)

        # update the agent using realized reward + bandit learning
        self.agent.update_observation(observation, action, reward)

        # log whatever we need for the plots we will want to use
        instant_regret = optimal_reward - expected_reward
        self.cum_regret += instant_regret

        # advance the environment (used in nonstationary experiment)
        self.environment.advance(action, reward)

        if (t + 1) % self.rec_freq == 0:
            self.data_dict = {
                't': (t + 1),
                'instant_regret': instant_regret,
                'cum_regret': self.cum_regret,
                'action': action,
                'unique_id': self.unique_id
            }
            self.results.append(self.data_dict)

    def run_experiment(self):
        """Run the experiment for n_steps and collect data."""
        np.random.seed(self.seed)
        self.cum_regret = 0
        self.com_optimal = 0

        for t in range(self.n_steps):
            self.run_step_maybe_log(t)

        self.results = pd.DataFrame(self.results)


class ExperimentWithMean(BaseExperiment):
    def run_step_maybe_log(self, t):
        # evolve the bandit (potentially contetual) for one step and action
        observation = self.environment.get_observation()
        action = self.agent.pick_action(observation)

        # compute useful stuff for regret calculations
        optimal_reward = self.environment.get_optimal_reward()
        expected_reward = self.environment.get_expected_reward(action)
        reward = self.environment.get_expected_reward(action)

        # update the agent using realized rewards + bandit learning
        self.agent.update_observation(observation, action, reward)

        # log whatever we need for the plots we will want to use.
        instant_regret = optimal_reward - expected_reward
        self.cum_regret += instant_regret

        # advance the environment (used in nonstationary experiment)
        self.environment.advance(action, reward)

        if (t + 1) % self.rec_freq == 0:
            self.data_dict = {
                't': (t + 1),
                'instant_regret': instant_regret,
                'cum_regret': self.cum_regret,
                'posterior_mean': self.agent.get_posterior_mean(),
                'unique_id': self.unique_id
            }
            self.results.append(self.data_dict)


class ExperimentNoAction(BaseExperiment):
    def run_step_maybe_log(self, t):
        # evolve the bandit (potentially contextual) for one step and pick action
        observation = self.environment.get_observation()
        action = self.agent.pick_action(observation)

        # compute useful stuff for regret calculations
        optimal_reward = self.environment.get_optimal_reward()
        expected_reward = self.environment.get_expected_reward(action)
        reward = self.environment.get_stochastic_reward(action)

        # update the agent using reward + bandit learning
        self.agent.update_observation(observation, action, reward)

        # log whatever we need for the plots we will want to use
        instant_regret = optimal_reward - expected_reward
        self.cum_optimal += optimal_reward
        self.cum_regret += instant_regret

        # advance the environment (used in nonstationary experiment)
        self.environment.advance(action, reward)

        if (t + 1) % self.rec_freq == 0:
            self.data_dict = {
                't': (t + 1),
                'instant_regret': instant_regret,
                'cum_regret': self.cum_regret,
                'cum_optimal': self.cum_optimal,
                'unique_id': self.unique_id
            }
            self.results.append(self.data_dict)
