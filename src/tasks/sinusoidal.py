"""Module that defines the sinusoidal tracking tasks."""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from tasks.base_task import BaseTask


class BaseSine(BaseTask, ABC):
    def __init__(
        self,
        env,
        period: float,
        amplitude: float,
    ) -> None:
        """Task to track a sinusoidal reference signal.

        Args:
            env: The environment.
            period: The period of the sinusoidal signal [s].
            amplitude: The amplitude of the sinusoidal signal.
        """
        super().__init__(env)
        self.period = period
        self.amplitude = amplitude

    def reference(self):
        """The reference signal."""
        period = self.period
        amplitude = self.amplitude

        reference = amplitude * np.sin(2 * np.pi / period * self.env.current_time)

        return np.array([reference])


class SineQ(BaseSine):
    """Task to track a sinusoidal reference signal of the pitch rate q."""

    tracked_states = ["q"]

    def __init__(self, env):
        super().__init__(env, period=3, amplitude=0.1)

    def __str__(self):
        return "sin_q"


class SineTheta(BaseSine):
    """Task to track a sinusoidal reference signal of the pitch angle theta."""

    tracked_states = ["theta"]

    def __init__(self, env):
        super().__init__(env, period=3, amplitude=0.1)

    def __str__(self):
        return "sin_theta"
