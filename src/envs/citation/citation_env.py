"""Creates a gym environment for the high fidelity citation  model."""
import numpy as np
from gymnasium import spaces

from envs.base_env import BaseEnv
from envs.citation.models.model_loader import load_model


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
    ):
        self.model = load_model(model)
        if tracked_state not in self.model.states:
            raise ValueError(
                f"Tracked state {tracked_state} not in model states {self.model.states}"
            )

        self.model.initialize()

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
        return spaces.Box(
            low=-0.3, high=0.3, shape=(self.model.n_inputs,), dtype=np.float32
        )

    def state_transition(self, action):
        """The state transition function of the environment."""
        states = self.model.step(action.flatten())
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
