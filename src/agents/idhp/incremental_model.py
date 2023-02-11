"""IDHP incremental model."""
import numpy as np
from abc import ABC
import gym

from envs.lti_citation.aircraft_environment import AircraftEnv
from thesis_pilot.config import ConfigLinearAircraft


class IncrementalModelBase(ABC):
    """Base class for incremental envs based on gym environmnet.

    Attributes:
        env: gym environment
        n_inputs: Number of inputs to the model.
        n_states: Number of states in the model.
        gamma: Discount factor.
        F: State transition matrix.
        G: Input matrix.
        actions: List of actions taken.
        states: List of states visited.
        cov: Covariance matrix.
        theta: Parameter matrix.
    """

    def __init__(self,
                 env: gym.Env,
                 n_inputs: int,
                 n_states: int,
                 gamma: float = 0.8, ) -> None:
        """Initialize the incremental model.

        Args:
            env: gym environment
            n_inputs: Number of inputs to the model.
            n_states: Number of states in the model.
            gamma: Discount factor.


        """
        self.env = env
        self.n_inputs = n_inputs
        self.n_states = n_states
        self.gamma = gamma

        self.actions = []
        self.states = []

        # Initialize covariance matrices
        self.cov = np.eye(self.n_states + self.n_inputs)

        # Initialize theta
        self.theta = np.zeros((self.n_states + self.n_inputs, self.n_states))

    def update(self) -> None:
        """Update the model's covariance and parameter matrices."""
        # Only start updating after the first step
        if len(self.actions) < 2:
            return

        # Get the change in action and state
        u_k, u_k1 = self.actions[-2:]
        x_k, x_k1 = self.states[-2:]

        du = u_k1 - u_k
        dx = x_k1 - x_k

        # Build the input information matrix
        X = np.vstack((dx, np.array(du)))

        # Get the predicted change in state
        dx_hat = X.T @ self.theta

        # Get the error
        e = dx.T - dx_hat

        # Define matrices used to update theta and cov
        cov_at_x = self.cov @ X
        x_at_cov = X.T @ self.cov
        x_at_cov_at_x = x_at_cov @ X

        # Update the parameter matrix theta
        theta_k1 = self.theta + (cov_at_x @ np.linalg.inv(self.gamma + x_at_cov_at_x)) * e
        cov_k1 = (1 / self.gamma) * (self.cov - (cov_at_x @ x_at_cov) / (self.gamma + x_at_cov_at_x))

        self.theta, self.cov = theta_k1, cov_k1

    @property
    def F(self) -> np.ndarray:
        """Return the state transition matrix."""
        return self.theta[:self.n_states, :].T

    @property
    def G(self) -> np.ndarray:
        """Return the input matrix."""
        return self.theta[self.n_states:, :].T


class IncrementalLTIAircraft(IncrementalModelBase):
    """Incremental model for the LTI aircraft system."""

    def __init__(self, _env: AircraftEnv) -> None:
        """Initialize the model."""
        n_inputs = _env.aircraft.ss.ninputs
        n_states = _env.aircraft.ss.nstates

        super(IncrementalLTIAircraft, self).__init__(_env, n_inputs, n_states)

    def learn(self):
        action = 1

        # Initial condition
        self.actions.append(0)
        self.states.append(self.env.aircraft.current_state)
        for i in range(10):
            observation, reward, done, info = self.env.step(0.1)

            self.actions.append(action)
            self.states.append(self.env.aircraft.current_state)

            if len(self.actions) > 1:
                self.update()


env = AircraftEnv(ConfigLinearAircraft(configuration="sp"))

im = IncrementalLTIAircraft(env)
im.learn()
