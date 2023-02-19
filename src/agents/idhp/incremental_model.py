"""IDHP incremental model."""
import numpy as np
from abc import ABC
from envs.lti_citation.aircraft_environment import AircraftEnv


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
                 n_states,
                 n_inputs: int,
                 gamma: float = 0.8, ) -> None:
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

    def update(self, future_state) -> None:
        """Update the model's covariance and parameter matrices.

        Args:
            future_state: The state from the environment after the current action is taken.
        """
        # Only start updating after the first step
        if self.action_k_1 is None and self.state_k_1 is None:
            return

        # Get the predicted change in state
        dx_hat, X = self.predict(self.state_k, self.state_k_1,
                                 self.action_k, self.action_k_1,
                                 increment=True)

        dx = future_state - self.state_k

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

    def predict(self, state: np.ndarray, state_before: np.ndarray, action: np.ndarray, action_before: np.ndarray,
                increment=False) -> np.ndarray:
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
        X = np.vstack((dx, np.array(du)))

        # Get the predicted change in state
        dx_hat = X.T @ self.theta

        return (dx_hat, X) if increment else state + dx_hat


class IncrementalLTIAircraft(IncrementalModelBase):
    """Incremental model for the LTI aircraft system."""

    def __init__(self, env: AircraftEnv) -> None:
        """Initialize the model."""
        n_states = env.aircraft.ss.nstates
        n_inputs = env.aircraft.ss.ninputs
        super(IncrementalLTIAircraft, self).__init__(n_states, n_inputs)

    def increment(self, env: AircraftEnv, action: float) -> None:
        """Increment the model."""
        self.action_k_1 = self.action_k
        self.action_k = np.array(action)

        self.state_k_1 = self.state_k
        self.state_k = env.aircraft.current_state

    def update(self, env: AircraftEnv):
        """Update the model."""
        super(IncrementalLTIAircraft, self).update(env.aircraft.current_state)
