"""Module that defines the attitude tracking tasks."""
from abc import ABC

import numpy as np

from helpers.signals import cos_step
from tasks.base_task import BaseTask


class Attitude(BaseTask, ABC):
    """Task to track a reference signal of the attitude."""

    tracked_states = ["theta", "phi", "beta"]

    def __init__(self, env):
        super().__init__(env)

    @property
    def scale(self):
        """The scale of each state tracked in the task."""
        return 1 / np.deg2rad([30, 30, 7])


class AttitudeTrain(Attitude):
    """Task to train the attitude controller."""

    def __init__(self, env):
        super().__init__(env)
        self.amp_theta = np.deg2rad(
            np.random.choice([20, 10, -10, -20], 1)
        )  # amplitude [rad]
        self.amp_phi = np.deg2rad(
            np.random.choice([40, 20, -20, -40], 1)
        )  # amplitude [rad]

    def __str__(self):
        return "att_train"

    def reference(self) -> np.ndarray:
        """Reference signal."""
        t = self.env.current_time

        # Theta reference
        amp_theta = self.amp_theta  # amplitude [rad]
        theta_ref = amp_theta * (
            cos_step(t, 0.25, 1) - cos_step(t, 0.5, 1) + cos_step(t, 0.75, 1)
        )

        # Phi reference
        amp_phi = self.amp_phi  # amplitude [rad]
        phi_ref = amp_phi * (
            cos_step(t, 0.25, 1) - cos_step(t, 0.5, 1) + cos_step(t, 0.75, 1)
        )
        # phi_ref = amp_phi * (cos_step(t, 0.25, 1) - cos_step(t, 4, 1))

        # Beta reference
        beta_ref = 0

        return np.hstack((theta_ref, phi_ref, beta_ref))


class SineAttitude(Attitude):
    """Task to track a sinusoidal reference signal of the citation atittude."""

    def __init__(self, env):
        super().__init__(env)
        self.amp_theta = np.deg2rad(
            np.random.choice([20, 10, -10, -20], 1)
        )  # amplitude [rad]
        self.amp_phi = np.deg2rad(
            np.random.choice([40, 20, -20, -40], 1)
        )  # amplitude [rad]

    def __str__(self):
        return "sin_att"

    def reference(self) -> np.ndarray:
        """Reference signal."""
        t = self.env.current_time
        period = 3  # s

        # Theta reference
        amp_theta = self.amp_theta  # amplitude [rad]
        theta_ref = amp_theta * np.sin(2 * np.pi / period * t)

        # Phi reference
        amp_phi = self.amp_phi  # amplitude [rad]
        phi_ref = amp_phi * np.sin(2 * np.pi / period * t)

        # Beta reference
        beta_ref = 0

        return np.hstack((beta_ref, phi_ref, beta_ref))


class SinAttitudeEvaluate(SineAttitude):
    def __init__(self, env):
        super().__init__(env)
        self.amp_phi = np.deg2rad(20)  # amplitude [rad]
        self.amp_theta = np.deg2rad(10)  # amplitude [rad]

    def __str__(self):
        return "sin_att_eval"
