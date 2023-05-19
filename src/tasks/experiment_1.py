"""Tasks for experiment 1 of the report that compares the performance on different reference signals."""
import numpy as np

from tasks.attitude import Attitude


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
        amp_theta = np.deg2rad(20)  # amplitude [rad]
        theta_ref = amp_theta * np.sin(2 * np.pi * 0.25 * t)

        # Phi reference
        amp_phi = np.deg2rad(40)  # amplitude [rad]
        phi_ref = amp_phi * np.sin(2 * np.pi * 0.25 * t)

        # Beta reference
        beta_ref = 0

        return np.hstack((theta_ref, phi_ref, beta_ref))
