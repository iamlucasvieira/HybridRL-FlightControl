"""Module that contains tests for the SAC buffer."""

import pytest
from agents.sac.buffer import ReplayBuffer


class TestBuffer:
    """Test the SAC buffer."""

    def test_init(self):
        """Test the initialization of the buffer."""
        buffer = ReplayBuffer(10)
        assert buffer.len == 0
        assert buffer.size == 10

    def test_append(self):
        """Test the append method of the buffer."""
        buffer = ReplayBuffer(10)
        buffer.append((1, 2, 3, 4, 5))
        assert buffer.len == 1
        assert buffer[0].state == 1
        assert buffer[0].action == 2
        assert buffer[0].reward == 3
        assert buffer[0].state_ == 4
        assert buffer[0].done == 5

    def test_multiple_append(self):
        """Test the append method of the buffer."""
        buffer = ReplayBuffer(10)
        for i in range(10):
            buffer.append((1, 2, 3, 4, 5))
            assert buffer.len == i + 1

    def test_full_buffer(self):
        """Test the append method of the buffer."""
        buffer = ReplayBuffer(10)
        for i in range(10):
            buffer.append((1, 2, 3, 4, 5))
        assert buffer.full()

        buffer.append((1, 2, 3, 4, 5))
        assert buffer.len == 10

    @pytest.mark.parametrize('batch_size', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_sample_buffer(self, batch_size):
        """Test the sample buffer method."""
        buffer = ReplayBuffer(10)
        for i in range(10):
            buffer.append((i, i, i, i, i))
        states, actions, rewards, states_, dones = buffer.sample_buffer(batch_size)
        assert len(states) == batch_size
        assert len(actions) == batch_size
        assert len(rewards) == batch_size
        assert len(states_) == batch_size
        assert len(dones) == batch_size

    def test_sample_buffer_empty(self):
        """Test the sample buffer method."""
        buffer = ReplayBuffer(10)
        assert buffer.empty()

        with pytest.raises(ValueError):
            buffer.sample_buffer(1)

    def test_sample_buffer_batch_size_zero(self):
        """Test the sample buffer method."""
        buffer = ReplayBuffer(10)
        for i in range(10):
            buffer.append((i, i, i, i, i))

        with pytest.raises(ValueError):
            buffer.sample_buffer(0)

    def test_sample_buffer_batch_size_too_large(self):
        """Test the sample buffer method."""
        buffer = ReplayBuffer(10)
        batch_size = 11
        buffer_size = 10
        for i in range(buffer_size):
            buffer.append((i, i, i, i, i))

        states, actions, rewards, states_, dones = buffer.sample_buffer(batch_size)
        assert len(states) == buffer_size
        assert len(actions) == buffer_size
        assert len(rewards) == buffer_size
        assert len(states_) == buffer_size
        assert len(dones) == buffer_size
