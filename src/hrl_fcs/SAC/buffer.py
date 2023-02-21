"""Creates SAC algorithm.

Source: https://www.youtube.com/watch?v=ioidsRlf79o
"""
import numpy as np


class ReplayBuffer():
    """Replay buffer for storing and sampling experiences of the SAC algorithm."""

    def __init__(self, max_size, input_shape, n_actions):
        """Initialize replay buffer.

        args:
            max_size: Maximum size of the replay buffer.
            input_shape: Shape of the input.
            n_actions: Number of actions.
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        """Store transition in replay buffer.

        args:
            state: Current state.
            action: Current action.
            reward: Current reward.
            state_: Next state.
            done: Done flag.
        """
        # Get index of the current memory counter
        index = self.mem_cntr % self.mem_size

        # Saves the transition in the replay buffer
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones