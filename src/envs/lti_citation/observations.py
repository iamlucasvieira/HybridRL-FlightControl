"""Module that defines the possible observation vectors."""
import numpy as np


def get_observation(observation_type):
    """Returns the reward function."""
    if observation_type not in observations_dict:
        raise ValueError(f"Observation {observation_type} not found.")
    else:
        return observations_dict[observation_type]


def get_value(item, idx=-1):
    min_length = idx + 1 if idx >= 0 else abs(idx)
    if isinstance(item, list) and len(item) >= min_length:
        return item[idx]
    else:
        return 0

    return 0 if not item else item[idx]


class Observations:
    """Class that contains the reward functions."""

    @staticmethod
    def states_ref_error(self):
        """Returns the all states, the reference state, and the tracking error."""
        return np.append(
            self.x_t,
            [np.array([self.x_t_r]),
             get_value(self.sq_error)]).astype(np.float32)

    @staticmethod
    def error(self):
        """Returns the squared error."""
        return np.array([get_value(self.sq_error)]).astype(np.float32)

    @staticmethod
    def ref_state(self):
        """Returns the reference and the value of the tracked state"""
        return np.array([self.x_t_r, get_value(self.track)]).astype(np.float32)

    @staticmethod
    def state_error(self):
        """Returns the tracked state and the tracking error"""
        return np.array([get_value(self.track), get_value(self.sq_error)]).astype(np.float32)

    @staticmethod
    def states(self):
        """Returns the all states"""
        return np.array(self.x_t).astype(np.float32)

    @staticmethod
    def states_ref(self):
        """Returns the all states and the reference state"""
        return np.append(
            self.aircraft.current_state,
            [get_value(self.reference, idx=-2)]).astype(np.float32)


observations_dict = {
    "states + ref + error": Observations.states_ref_error,
    "error": Observations.error,
    "ref + state": Observations.ref_state,
    "state + error": Observations.state_error,
    "states": Observations.states,
    "states + ref": Observations.states_ref,
}

AVAILABLE_OBSERVATIONS = list(observations_dict.keys())
