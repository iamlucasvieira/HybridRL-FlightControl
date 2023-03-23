"""Module that defines the base policy class."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from gymnasium import spaces
from torch import nn


spaces_type_alias = Union[spaces.Box, spaces.Space]


class BasePolicy(nn.Module, ABC):
    """Base policy class."""

    def __init__(
        self,
        observation_space: spaces_type_alias,
        action_space: spaces_type_alias,
        _init_setup_policy: bool = True,
    ):
        """Initialize policy.

        args:
            observation_space: Observation space.
            action_space: Action space.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        if _init_setup_policy:
            self._setup_policy()

    @abstractmethod
    def _setup_policy(self):
        pass

    @abstractmethod
    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Predict action."""
        pass
