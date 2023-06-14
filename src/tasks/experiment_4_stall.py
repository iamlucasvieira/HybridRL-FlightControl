from tasks.attitude import Attitude

import numpy as np

class StallTask(Attitude):

    def __str__(self):
        return "stall"

    def reference(self) -> np.ndarray:
        """Reference signal."""
        t = self.env.current_time

        # Theta reference
        theta_ref = 10

        # Phi reference
        phi_ref = 0

        # Beta reference
        beta_ref = 0

        return np.hstack((theta_ref, phi_ref, beta_ref))