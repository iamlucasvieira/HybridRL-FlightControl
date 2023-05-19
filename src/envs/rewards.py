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
        sq_error = np.sum(self.sq_error[-1])
        return -self.reward_scale * sq_error

    @staticmethod
    def sq_error_da(self):
        """Returns the squared error between the reference and the state."""
        return -(
            self.sq_error[-1] + 0.1 * np.sum(np.abs(np.diff(self.actions[-2:], axis=0)))
        )

    @staticmethod
    def sq_error_da_a(self):
        """Returns the squared error, the action difference and the action"""
        return -(
            self.sq_error[-1]
            + 0.1 * np.sum(np.abs(np.diff(self.actions[-2:], axis=0)))
            + 0.1 * np.sum(np.abs(self.actions[-1]))
        )

    @staticmethod
    def error(self):
        """Returns the error between the reference and the state."""
        return self.reward_scale * self.error[-1]

    @staticmethod
    def clip(self):
        """Reward that clips the reward to -1, 1"""
        n_tracked_states = self.task.mask.sum()
        reward = -1 / n_tracked_states * np.abs(np.clip(self.error[-1], -1, 1)).sum()

        return reward


rewards_dict = {
    "sq_error": Rewards.sq_error,
    "sq_error_da": Rewards.sq_error_da,
    "sq_error_da_a": Rewards.sq_error_da_a,
    "error": Rewards.error,
    "clip": Rewards.clip,
}

AVAILABLE_REWARDS = list(rewards_dict.keys())
