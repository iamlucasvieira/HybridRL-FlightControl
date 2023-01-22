"""Module that contains the reward functions for the environment."""

import numpy as np


def get_reward(reward_type):
    """Returns the reward function."""
    if reward_type not in rewards_dict:
        raise ValueError(f"Reward {reward_type} not found.")
    else:
        return rewards_dict[reward_type]

class Rewards:
    """Class that contains the reward functions."""

    @staticmethod
    def sq_error(self):
        """Returns the squared error between the reference and the state."""
        return - self.reward_scale * self.sq_error[-1]

    @staticmethod
    def sq_error_da(self):
        """Returns the squared error between the reference and the state."""
        return -(self.sq_error[-1] + 0.1*np.sum(np.abs(np.diff(self.actions[-2:], axis=0))))

    @staticmethod
    def sq_error_da_a(self):
        """Returns the squared error, the action difference and the action"""
        return -(self.sq_error[-1] + 0.1*np.sum(np.abs(np.diff(self.actions[-2:], axis=0))) + 0.1*np.sum(np.abs(self.actions[-1])))


rewards_dict = {
    "sq_error": Rewards.sq_error,
    "sq_error_da": Rewards.sq_error_da,
    "sq_error_da_a": Rewards.sq_error_da_a,
}

AVAILABLE_REWARDS = list(rewards_dict.keys())