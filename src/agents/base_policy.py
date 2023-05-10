"""Module that defines the base policy class."""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from helpers.torch_helpers import get_device, to_tensor

spaces_type_alias = Union[spaces.Box, spaces.Space]


class BasePolicy(nn.Module, ABC):
    """Base policy class."""

    def __init__(
            self,
            observation_space: spaces_type_alias,
            action_space: spaces_type_alias,
            _init_setup_policy: bool = True,
            device: Optional[str] = None,
    ):
        """Initialize policy.

        args:
            observation_space: Observation space.
            action_space: Action space.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        if device is None:
            device = get_device()
        self.device = device

        if _init_setup_policy:
            self._setup_policy()
            self.to(self.device)

    @abstractmethod
    def _setup_policy(self):
        pass

    @abstractmethod
    def _predict(
            self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """Predict action."""
        pass

    def predict(
            self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """Predict action."""
        action = self._predict(observation, deterministic)
        unscaled_action = self.unscale_action(action)
        return unscaled_action

    def scale_action(self, action: np.ndarray):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high

        if isinstance(scaled_action, th.Tensor):
            low, high = to_tensor(low, high, device=self.device)

        return low + (0.5 * (scaled_action + 1.0) * (high - low))
