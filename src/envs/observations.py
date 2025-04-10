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
        return (
            np.hstack((self.states[-1], self.reference[-1], self.sq_error[-1]))
            .astype(np.float32)
            .flatten()
        )

    @staticmethod
    def error(self):
        """Returns the squared error."""
        return self.sq_error[-1].astype(np.float32).flatten()

    @staticmethod
    def ref_state(self):
        """Returns the reference and the value of the tracked state"""
        return (
            np.hstack((self.reference[-1], self.track[-1])).astype(np.float32).flatten()
        )

    @staticmethod
    def state_error(self):
        """Returns the tracked state and the tracking error"""
        return (
            np.hstack((self.track[-1], self.sq_error[-1])).astype(np.float32).flatten()
        )

    @staticmethod
    def states(self):
        """Returns the all states"""
        return self.states[-1].astype(np.float32).flatten()

    @staticmethod
    def states_ref(self):
        """Returns the all states and the reference state"""
        return (
            np.hstack((self.states[-1], self.reference[-1]))
            .astype(np.float32)
            .flatten()
        )

    @staticmethod
    def states_error(self):
        """Returns the all states and the reference state"""
        return np.hstack((self.states[-1], self.error[-1])).astype(np.float32).flatten()

    @staticmethod
    def noise_states_ref(self):
        """Return the all states and the reference state with noise"""
        bias = 5e-3
        variance = 1e-3
        noise = np.random.normal(bias, variance, self.states[-1].shape)
        return (
            np.hstack((self.states[-1] + noise, self.reference[-1]))
            .astype(np.float32)
            .flatten()
        )

    @staticmethod
    def sac_attitude(self):
        """Citation observation for sac attitude tracking"""
        obs_states = ["p", "q", "r"]
        states_idx = [self.states_name.index(s) for s in obs_states]

        return np.hstack((self.states[-1][states_idx], self.error[-1]))

    @staticmethod
    def sac_attitude_noise(self):
        """Citation observation for sac attitude tracking"""
        obs_states = ["p", "q", "r"]
        states_idx = [self.states_name.index(s) for s in obs_states]

        noise_bias = 3e-5
        noise_variance = 4e-7
        noise = np.random.normal(noise_bias, noise_variance, len(states_idx))

        return np.hstack((self.states[-1][states_idx] + noise, self.error[-1]))

    @staticmethod
    def idhp_citation(self):
        """Citation observation for IDHP attitude tracking."""
        obs_states = ["p", "q", "r", "alpha", "theta", "phi", "beta"]
        states_idx = [self.states_name.index(s) for s in obs_states]

        return np.hstack((self.states[-1][states_idx], self.error[-1]))


observations_dict = {
    "states + ref + error": Observations.states_ref_error,
    "error": Observations.error,
    "ref + state": Observations.ref_state,
    "state + error": Observations.state_error,
    "states": Observations.states,
    "states + ref": Observations.states_ref,
    "states + error": Observations.states_error,
    "noise + states + ref": Observations.noise_states_ref,
    "sac_attitude": Observations.sac_attitude,
    "sac_attitude_noise": Observations.sac_attitude_noise,
    "idhp_citation": Observations.idhp_citation,
}

AVAILABLE_OBSERVATIONS = list(observations_dict.keys())
