"""Module that solves the cleaning robot RL problem."""
import numpy as np


class Robot:
    pass


class Room:
    def __init__(self):
        self.n = 15
        self.actions = [1, -1]
        self.board = np.zeros(self.n)
        self.policy = np.ones(self.n, dtype=int)

    def print_policy(self):
        """Print the policy."""
        policy = np.zeros(self.n, dtype=str)
        policy[self.policy == 1] = "→"
        policy[self.policy == -1] = "←"
        policy[0] = "X"
        policy[-1] = "x"

        return " ".join(policy)

    def state_transition(self, state, action):
        new_state = state + action
        if state == 0 or state == self.n - 1:
            reward = 0
            new_state = state
        elif state == 1 and action == -1:
            reward = 10
        elif state == self.n - 2 and action == 1:
            reward = 5
        else:
            reward = 0

        return new_state, reward

    def bellman_equation(self, state, values):
        gamma = 0.9
        new_state, reward = self.state_transition(state, self.policy[state])
        new_value = reward + gamma * values[new_state]
        return new_value

    def policy_evaluation(self):
        """Evaluate the policy to determine the value function."""
        new_values = np.zeros(self.n)

        theta = 1e-6
        delta = np.inf

        while delta > theta:
            values = new_values.copy()

            for state in range(self.n):
                new_value = self.bellman_equation(state, values)
                new_values[state] = new_value

            delta = np.sum(np.abs(values - new_values))

        return new_values

    def policy_improvement(self, values):
        gamma = 0.9
        new_policy = np.zeros(self.n, dtype=int)
        for state in range(self.n):
            max_value = -np.inf
            for action in self.actions:
                new_state, reward = self.state_transition(state, action)
                new_value = reward + gamma * values[new_state]
                if new_value > max_value:
                    max_value = new_value
                    new_policy[state] = action
        return new_policy

    def policy_iteration(self, to_print=False):
        """Iterate over the policy evaluation and improvement."""
        counter = 0
        while True:
            counter += 1
            policy_str = self.print_policy()
            print(f"t_{counter:<5} {policy_str}")
            new_values = self.policy_evaluation()
            new_policy = self.policy_improvement(new_values)

            if np.array_equal(new_policy, self.policy):
                break
            else:
                self.policy = new_policy


if __name__ == "__main__":
    room = Room()
    room.policy_iteration(to_print=True)
