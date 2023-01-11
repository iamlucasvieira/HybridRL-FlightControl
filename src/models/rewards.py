"""Module that contains the reward functions for the environment."""

import numpy as np


def get_reward(reward_type):
    """Returns the reward function."""
    if reward_type == "sq_error":
        return Rewards.sq_error
    elif reward_type == "sq_error_da":
        return Rewards.sq_error_da
    elif reward_type == "sq_error_da_a":
        return Rewards.sq_error_da_a
    else:
        raise ValueError(f"Reward type {reward_type} not found")

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