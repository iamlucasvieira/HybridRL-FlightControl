"""Module that defines the possible observation vectors."""
import numpy as np


def get_observation(observation_type):
    """Returns the reward function."""
    if observation_type not in observations_dict:
        raise ValueError(f"Observation {observation_type} not found.")
    else:
        return observations_dict[observation_type]


class Observations:
    """Class that contains the reward functions."""

    @staticmethod
    def states_ref_error(self):
        """Returns the all states, the reference state, and the tracking error."""
        return np.hstack((
            self.states[-1],
            self.reference[-1],
            self.sq_error[-1])).astype(np.float32).flatten()

    @staticmethod
    def error(self):
        """Returns the squared error."""
        return self.sq_error[-1].astype(np.float32).flatten()

    @staticmethod
    def ref_state(self):
        """Returns the reference and the value of the tracked state"""
        return np.hstack((
            self.reference[-1],
            self.track[-1])).astype(np.float32).flatten()

    @staticmethod
    def state_error(self):
        """Returns the tracked state and the tracking error"""
        return np.hstack((
            self.track[-1],
            self.sq_error[-1])).astype(np.float32).flatten()

    @staticmethod
    def states(self):
        """Returns the all states"""
        return self.states[-1].astype(np.float32).flatten()

    @staticmethod
    def states_ref(self):
        """Returns the all states and the reference state"""
        return np.hstack((
            self.states[-1],
            self.reference[-1])).astype(np.float32).flatten()


observations_dict = {
    "states + ref + error": Observations.states_ref_error,
    "error": Observations.error,
    "ref + state": Observations.ref_state,
    "state + error": Observations.state_error,
    "states": Observations.states,
    "states + ref": Observations.states_ref,
}

AVAILABLE_OBSERVATIONS = list(observations_dict.keys())
