"""Module that defines the base task class."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseTask(ABC):
    """The base class for all tasks."""

    def __init__(self, env):
        self.env = env
        if not self.valid_for_env(env):
            raise ValueError(
                f"Task {self} is not valid for the environment {self.env}."
            )

    @property
    @abstractmethod
    def tracked_states(self) -> List[str]:
        """The states to be tracked."""
        ...

    def valid_for_env(self, env):
        """Check if the task is valid for the environment."""
        for state in self.tracked_states:
            if state not in env.states_name:
                return False
        return True

    @property
    def mask(self):
        """The mask for the tracked states."""
        # Get the indices of the tracked states
        idx_tracked_states = [
            self.env.states_name.index(state) for state in self.tracked_states
        ]

        # Create the mask
        mask = np.zeros(self.env.n_states, dtype=bool)
        mask[idx_tracked_states] = 1

        return mask

    @abstractmethod
    def __str__(self):
        """Return the name of the task."""
        ...

    @abstractmethod
    def reference(self) -> np.ndarray:
        """The reference signal."""
        ...

    @property
    @abstractmethod
    def scale(self) -> float:
        """The scale of each state tracked in the task."""
        ...
