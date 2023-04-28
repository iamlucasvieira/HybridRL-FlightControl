"""Creates a gym environment for the high fidelity citation  model."""
import numpy as np
from gymnasium import spaces
from collections import namedtuple
from envs.base_env import BaseEnv
from envs.citation.models.model_loader import load_model
from typing import List


class CitationEnv(BaseEnv):
    """Citation Environment that follows gym interface"""

    def __init__(
        self,
        model: str = "default",
        dt: float = 0.1,
        episode_steps: int = 100,
        reward_scale: float = 1.0,
        tracked_state: str = "q",
        reference_type: str = "sin",
        reward_type: str = "sq_error",
        observation_type: str = "states + ref + error",
        input_names: List[str] = None,
        observation_names: List[str] = None,
    ):
        self.model = load_model(model)
        self.input_names = ["de", "da", "dr"] if input_names is None else input_names
        self.observation_names = (
            ["p", "q", "r", "alpha", "beta", "phi", "theta"]
            if observation_names is None
            else observation_names
        )

        self.list_contains_all(self.model.states, [tracked_state])
        self.list_contains_all(self.model.inputs, self.input_names)
        self.list_contains_all(self.model.states, self.observation_names)

        self.model.initialize()

        self.input_idx = [self.model.inputs.index(name) for name in self.input_names]
        self.observation_idx = [
            self.model.states.index(name) for name in self.observation_names
        ]

        self._states = [
            self._initial_state().reshape(-1, 1)
        ]  # the .states stores the states flattened
        super().__init__(
            dt=dt,
            episode_steps=episode_steps,
            reward_scale=reward_scale,
            tracked_state=tracked_state,
            reference_type=reference_type,
            reward_type=reward_type,
            observation_type=observation_type,
        )

    @property
    def tracked_state_mask(self):
        """A mask that has the shape of the aircraft states and the value 1 in the tracked state."""
        state_mask = np.zeros(self.model.n_states)
        tracked_state_index = self.model.states.index(self.tracked_state)
        state_mask[tracked_state_index] = 1
        return state_mask.astype(bool)

    def _action_space(self):
        """The action space of the environment."""
        Action = namedtuple("Action", "low high")

        de = Action(-20.05, 14.90)
        da = Action(-37.24, 37.24)
        dr = Action(-21.77, 21.77)

        low_list, high_list = [], []
        for action in self.input_names:
            if action == "de":
                low, high = de.low, de.high
            elif action == "da":
                low, high = da.low, da.high
            elif action == "dr":
                low, high = dr.low, dr.high
            else:
                raise ValueError(
                    "Only 'de', 'da', and 'dr' actions are supported by te model."
                )
            low_list.append(low)
            high_list.append(high)

        return spaces.Box(
            low=np.deg2rad(low_list),
            high=np.deg2rad(high_list),
            dtype=np.float32,
        )

    def state_transition(self, action):
        """The state transition function of the environment."""
        states = self.model.step(self.action_to_input(action).flatten())
        self._states.append(states.reshape(-1, 1))
        return states

    def _check_constraints(self, reward, done, info):
        return reward, done, info

    def _initial_state(self):
        """The initial state of the environment."""
        self._reset()
        initial_states = self.model.step(np.zeros(self.model.n_inputs))
        self._reset()
        return initial_states.flatten()

    def action_to_input(self, action):
        """Converts the action to the input of the model."""
        model_input = np.zeros(self.model.n_inputs)
        model_input[self.input_idx] = action
        return model_input

    def _reset(self):
        """Reset the environment."""
        self.model.terminate()
        self.model.initialize()

    @property
    def n_states(self) -> int:
        """The number of states in the environment."""
        return self.model.n_states

    @property
    def n_inputs(self) -> int:
        """The number of inputs in the environment."""
        return self.model.n_inputs

    @property
    def aircraft_states(self):
        """The states of the aircraft."""
        return self._states

    @property
    def current_aircraft_state(self):
        """The current state of the aircraft."""
        return self._states[-1]

    @staticmethod
    def list_contains_all(
        model_vars: List[str | int | float], variables: List[str | int | float]
    ):
        """Checks if a variable is in the model variables."""
        for var in variables:
            if var not in model_vars:
                raise ValueError(f"Variable {var} not in model variables {model_vars}")
        return True
