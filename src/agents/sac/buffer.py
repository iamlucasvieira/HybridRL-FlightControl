"""Creates a replay buffer for the SAC algorithm."""
import numpy as np
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'obs_', 'done'))


class ReplayBuffer(deque):
    """Replay buffer for storing and sampling experiences of the SAC algorithm."""

    def __init__(self, buffer_size: int):
        """Initialize replay buffer.

        args:
            buffer_size: Maximum size of the replay buffer.
        """
        super(ReplayBuffer, self).__init__(maxlen=buffer_size)

    @property
    def len(self) -> int:
        """Get the current size of the replay buffer."""
        return len(self)

    @property
    def size(self) -> int:
        """Get the maximum size of the replay buffer."""
        return self.maxlen

    def empty(self) -> bool:
        """Check if the replay buffer is empty."""
        return len(self) == 0

    def full(self) -> bool:
        """Check if the replay buffer is full."""
        return len(self) == self.maxlen

    def push(self, transition: tuple) -> None:
        """Append a new transition to the replay buffer."""
        super(ReplayBuffer, self).append(Transition(*transition))

    def append(self):
        """Raise an error if the user tries to append to the replay buffer."""
        raise NotImplementedError('Use push instead.')

    def ready(self, batch_size: int) -> bool:
        """Check if the replay buffer has enough samples to sample from."""
        return len(self) >= batch_size

    def sample_buffer(self, batch_size):
        """Sample a batch of transitions from the replay buffer."""
        if self.empty():
            raise ValueError('Cannot sample from an empty replay buffer.')
        elif batch_size == 0:
            raise ValueError('Batch size cannot be zero.')
        elif batch_size > len(self):
            batch_size = len(self)

        batch = random.sample(self, batch_size)
        return Transition(*zip(*batch))
