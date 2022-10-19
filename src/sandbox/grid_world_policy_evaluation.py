import numpy as np
from numpy.random import seed
from dataclasses import dataclass


class GridWorld:
    """Class that create the grid world problem."""

    def __init__(self):
        self.n = 5
        self.grid = np.zeros((self.n, self.n))
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.actions_symbols = ['↓', '↑', '→', '←']
        self.policy = np.ones((self.n, self.n, len(self.actions))) / len(self.actions)

    def print_policy(self):
        """Print the policy."""
        policy = np.zeros((self.n, self.n), dtype=str)
        for row in range(self.n):
            for column in range(self.n):
                idx_action = np.argmax(self.policy[row, column])
                policy[row, column] = self.actions_symbols[idx_action]

        print(policy)

    def state_transition(self, state, action):
        """Return the next state and reward."""
        state, action = np.array(state), np.array(action)

        if np.array_equal(state, (0, 1)):
            new_state = np.array((4, 1))
            reward = 10
        elif np.array_equal(state, (0, 3)):
            new_state = np.array((2, 3))
            reward = 5
        else:
            new_state = state + action
            reward = 0

            row, column = new_state

            if row < 0 or row >= self.n or column < 0 or column >= self.n:
                new_state = state
                reward = -1

        return new_state, reward

    def bellman_equation(self, state, value):
        """Return the value of the state using the Bellman equation."""
        gamma = 0.9
        new_value = 0
        for idx_action, action in enumerate(self.actions):
            new_state, reward = self.state_transition(state, action)
            new_value += self.policy[tuple(new_state) + (idx_action,)] * (reward + gamma * value[tuple(new_state)])

        return new_value

    def policy_evaluation(self):
        """Evaluates the policy by estimating the value function."""

        theta = 1e-6
        delta = np.inf

        new_values = np.zeros((self.n, self.n))

        while delta > theta:

            values = new_values.copy()

            for row in range(self.n):
                for column in range(self.n):
                    state = (row, column)
                    v_new = self.bellman_equation(state, values)
                    new_values[row, column] = v_new

            delta = np.sum(np.abs(values - new_values))

        return values

    def mc_policy_evalutation(self, n_episodes=100):
        """Evaluates the policy by Monte Carlo method."""
        seed(1)

        # events = np.zeros(n_episodes)
        n_events = 100

        gamma = 0.9
        for _ in range(n_episodes):
            state = np.random.randint(0, self.n, size=2)
            action = np.random.choice(self.actions, p=self.policy[tuple(state)])
            events = np.zeros(n_events)

            events[0] = Episode(state=state, action=action, reward=None)

            for i in range(1, n_events):
                state, reward = self.state_transition(state, action)

                action = np.random.choice(self.actions, p=self.policy[tuple(state)])

            # gain = 0
            # values =
            # for _ in range(100):
            #     action = np.random.choice(self.actions, p=self.policy[tuple(state)])
            #     state, reward = self.state_transition(state, action)
            #
            #     gain = gain * gamma + reward

        print(2)

    def policy_improvement(self, values):
        """Improve the policy by maximising the action value function."""
        for row in range(self.n):
            for column in range(self.n):
                state = (row, column)
                action_values = np.zeros(len(self.actions))
                for iddx_action, action in enumerate(self.actions):
                    new_state, reward = self.state_transition(state, action)
                    action_values[idx_action] = reward + values[tuple(new_state)]

                best_action = np.argmax(action_values)
                self.policy[row, column] = np.eye(len(self.actions))[best_action]

    def learn(self):
        """Learn the optimal policy."""
        values = self.policy_evaluation()
        self.policy_improvement(values)


@dataclass
class Episode:
    """Class that represents an episode."""
    state: float
    action: float
    reward: float


if __name__ == "__main__":
    grid_world = GridWorld()
    value = grid_world.policy_evaluation()
    print(value)
    value_mc = grid_world.mc_policy_evalutation()
    print(value_mc)
