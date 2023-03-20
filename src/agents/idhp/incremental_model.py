"""IDHP incremental model."""
from abc import ABC

import numpy as np

from envs import BaseEnv


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

    def __init__(
        self,
        n_states,
        n_inputs: int,
        gamma: float = 0.8,
    ) -> None:
        """Initialize the incremental model.

        Args:
            env: gym environment
            n_inputs: Number of inputs to the model.
            n_states: Number of states in the model.
            gamma: Discount factor.


        """
        self.n_inputs = n_inputs
        self.n_states = n_states
        self.gamma = gamma

        # Current action and state
        self.action_k = None
        self.state_k = None

        # Previous action and state
        self.action_k_1 = None
        self.state_k_1 = None

        # Initialize covariance matrices
        self.cov = np.eye(self.n_states + self.n_inputs)

        # Initialize theta
        self.theta = np.zeros((self.n_states + self.n_inputs, self.n_states))

        # Data
        self.errors = [np.array([0] * self.n_states)]

    @property
    def ready(self):
        """Return if the model is ready to be update."""
        READY = (
            (self.action_k is not None)
            and (self.state_k is not None)
            and (self.action_k_1 is not None)
            and (self.state_k_1 is not None)
        )
        return READY

    def update(self, future_state) -> None:
        """Update the model's covariance and parameter matrices.

        Args:
            future_state: The state from the environment after the current action is taken.
        """
        # Get the predicted change in state
        dx_hat, X = self.predict(
            self.state_k, self.state_k_1, self.action_k, self.action_k_1, increment=True
        )

        dx = future_state - self.state_k

        # Get the error
        e = dx.T - dx_hat.T
        self.errors.append(e)

        # Define matrices used to update theta and cov
        cov_at_x = self.cov @ X
        x_at_cov = X.T @ self.cov
        x_at_cov_at_x = x_at_cov @ X

        # Update the parameter matrix theta
        theta_k1 = self.theta + (cov_at_x / (self.gamma + x_at_cov_at_x)) * e
        cov_k1 = (1 / self.gamma) * (
            self.cov - (cov_at_x @ x_at_cov) / (self.gamma + x_at_cov_at_x)
        )

        self.theta, self.cov = theta_k1, cov_k1

    @property
    def F(self) -> np.ndarray:
        """Return the state transition matrix."""
        return self.theta[: self.n_states, :].T

    @property
    def G(self) -> np.ndarray:
        """Return the input matrix."""
        return self.theta[self.n_states :, :].T

    def predict(
        self,
        state: np.ndarray,
        state_before: np.ndarray,
        action: np.ndarray,
        action_before: np.ndarray,
        increment=False,
    ) -> np.ndarray:
        """Predict the next state given the current state and action.

        Args:
            state: Current state.
            state_before: Previous state.
            action: Current action.
            action_before: Previous action.
            increment: Whether to return the prediction or the increment, rather than the predicted state.
        """
        # Get the change in action and state
        u_k_1, u_k = action_before, action
        x_k_1, x_k = state_before, state

        du = u_k - u_k_1
        dx = x_k - x_k_1

        # Build the input information matrix
        X = np.vstack((dx, np.array(du).reshape(-1, 1)))

        # Get the predicted change in state
        dx_hat_t = X.T @ self.theta
        dx_hat = dx_hat_t.T
        return (dx_hat, X) if increment else state + dx_hat


class IncrementalCitation(IncrementalModelBase):
    """Incremental model for the LTI aircraft system."""

    def __init__(self, env: BaseEnv, **kwargs) -> None:
        """Initialize the model."""
        n_states = env.n_states
        n_inputs = env.n_inputs
        super(IncrementalCitation, self).__init__(n_states, n_inputs, **kwargs)

    def increment(self, env: BaseEnv) -> None:
        """Increment the model."""
        if len(env.actions) < 2 and len(env.aircraft_states) < 2:
            raise ValueError("Not enough data to increment the model.")
        self.action_k_1 = env.actions[-2]
        self.action_k = env.actions[-1]

        self.state_k_1 = env.aircraft_states[-2]
        self.state_k = env.aircraft_states[-1]

    def update(self, env: BaseEnv) -> None:
        """Update the model."""
        # Only update if two data points are available
        if self.ready:
            super(IncrementalCitation, self).update(env.current_aircraft_state)
        self.increment(env)
