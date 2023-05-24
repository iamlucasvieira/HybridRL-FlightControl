"""Tasks for experiment 1 of the report that compares the performance on different reference signals."""
import numpy as np

from tasks.attitude import Attitude
from helpers.signals import cos_step


class FixedSineAttitude(Attitude):
    """Task to track a fixed reference signal of the attitude."""

    def __init__(self, env):
        super().__init__(env)
        self.amp_theta = np.deg2rad(5)  # amplitude [rad]
        self.amp_phi = np.deg2rad(5)  # amplitude [rad]

    def __str__(self):
        return "fixed_sin_att"

    def reference(self) -> np.ndarray:
        """Reference signal."""
        t = self.env.current_time

        # Theta reference
        theta_ref = self.amp_theta * np.sin(2 * np.pi * 0.25 * t)

        # Phi reference
        phi_ref = self.amp_phi * np.sin(2 * np.pi * 0.25 * t)

        # Beta reference
        beta_ref = 0

        return np.hstack((theta_ref, phi_ref, beta_ref))


class PseudoRandomSine(Attitude):
    """Task to track a fixed reference signal of the attitude."""

    def __init__(self, env):
        super().__init__(env)
        self.amp_theta = np.deg2rad(5)  # amplitude [rad]
        self.amp_phi = np.deg2rad(5)  # amplitude [rad]

    def __str__(self):
        return "pseudo_random_sin_att"

    def reference(self) -> np.ndarray:
        """Reference signal."""
        t = self.env.current_time

        # Theta reference
        theta_ref = 0

        # Phi reference
        phi_ref = self.amp_phi * (np.sin(2 * np.pi * 1 / 4 * t) +
                                  np.cos(2 * np.pi * 1 / 6 * t) +
                                  np.sin(2 * np.pi * 1 / 8 * t)
                                  )

        # Beta reference
        beta_ref = 0

        return np.hstack((theta_ref, phi_ref, beta_ref))


class CossStep(Attitude):
    """Task to track a fixed reference signal of the attitude."""

    def __init__(self, env):
        super().__init__(env)
        self.amp_theta = np.deg2rad(5)
        self.amp_phi = np.deg2rad(5)

    def __str__(self):
        return "cos_step_att"

    def reference(self) -> np.ndarray:
        """Reference signal."""
        t = self.env.current_time

        # Theta reference
        theta_ref = self.amp_theta * (cos_step(t, 0.25, 2) -
                                      cos_step(t, 4, 2) +
                                      0.5 * cos_step(t, 8, 2) -
                                      0.5 * cos_step(t, 12, 2)
                                      )

        # Phi reference
        phi_ref = self.amp_phi * (0.3 * cos_step(t, 0.25, 3) -
                                  0.3 * cos_step(t, 4, 3) +
                                  cos_step(t, 9, 3) -
                                  cos_step(t, 15, 3)
                                  )

        # Beta reference
        beta_ref = 0

        return np.hstack((theta_ref, phi_ref, beta_ref))


class Hold(Attitude):
    """Task to track a fixed reference signal of the attitude."""

    def __init__(self, env):
        super().__init__(env)
        self.amp_theta = np.deg2rad(2)
        self.amp_phi = np.deg2rad(0)

    def __str__(self):
        return "hold_att"

    def reference(self) -> np.ndarray:
        """Reference signal."""
        t = self.env.current_time

        # Theta reference
        theta_ref = self.amp_theta

        # Phi reference
        phi_ref = self.amp_phi

        # Beta reference
        beta_ref = 0

        return np.hstack((theta_ref, phi_ref, beta_ref))
