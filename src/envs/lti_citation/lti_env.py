"""Creates a gym environment for the aircraft model"""
import numpy as np
from gymnasium import spaces

from envs.base_env import BaseEnv
from envs.lti_citation.models.lti_model import Aircraft


class LTIEnv(BaseEnv):
    def __init__(
        self,
        filename: str = "citation.yaml",
        configuration: str = "sp",
        dt: float = 0.1,
        episode_steps: int = 100,
        task_type: str = "sin_q",
        reward_type: str = "sq_error",
        observation_type: str = "states + ref + error",
        reward_scale: float = 1.0,
    ):
        self.aircraft = Aircraft(
            filename=filename,
            dt=dt,
            configuration=configuration,
        )
        self.input_names = ["de"]

        super().__init__(
            dt=dt,
            episode_steps=episode_steps,
            reward_scale=reward_scale,
            task_type=task_type,
            reward_type=reward_type,
            observation_type=observation_type,
        )

    def _action_space(self) -> spaces.Box:
        """The action space of the environment."""
        return spaces.Box(
            low=-0.3, high=0.3, shape=(self.aircraft.ss.ninputs,), dtype=np.float32
        )

    def state_transition(self, action):
        """The state transition function of the environment."""
        return self.aircraft.response(action).flatten()

    def _check_constraints(self, reward, done, info):
        """Check if the constraints are met."""
        if self.track[-1] > 0.5:
            reward *= 100
            done = True
            info = {"message": f"Reference too large: {self.track[-1]} > 0.5"}
        return reward, done, info

    def _initial_state(self):
        """The initial state of the environment."""
        # Reset the state of the environment to an initial state
        self.aircraft.build_state_space()
        x_t = self.aircraft.current_state.flatten()
        return x_t

    def _reset(self) -> None:
        """Reset the state of the environment to an initial state."""
        self.aircraft.build_state_space()

    @property
    def n_states(self) -> int:
        """The number of states in the environment."""
        return self.aircraft.ss.nstates

    @property
    def n_inputs(self) -> int:
        """The number of inputs in the environment."""
        return self.aircraft.ss.ninputs

    @property
    def aircraft_states(self):
        """The states of the aircraft."""
        return self.aircraft.states

    @property
    def current_aircraft_state(self):
        """The current state of the aircraft."""
        return self.aircraft.current_state

    @property
    def states_name(self):
        """The current state of the aircraft."""
        return self.aircraft.ss.x_names
