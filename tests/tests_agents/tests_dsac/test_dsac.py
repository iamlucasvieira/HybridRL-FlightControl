"""Module that tests the DSAC agent."""
from typing import Type

import pytest
import torch as th

from agents import DSAC
from agents.buffer import Transition
from agents.dsac.policy import DSACPolicy, generate_quantiles
from envs import BaseEnv, CitationEnv, LTIEnv


@pytest.fixture
def transition(env: Type[BaseEnv]) -> Transition:
    """Returns a transition."""
    env = env()
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return Transition(obs=obs, action=action, reward=reward, obs_=next_obs, done=done)


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestDSAC:
    """Class that tests the DSAC objct."""

    def test_init(self, env: Type[BaseEnv]):
        """Tests the initialization of the DSAC object."""
        agent = DSAC(env())
        assert isinstance(agent, DSAC)
        assert isinstance(agent.policy, DSACPolicy)
        assert isinstance(agent.target_policy, DSACPolicy)

    def test_huber_loss(self, env: Type[BaseEnv], transition):
        """Tests the huber loss of the DSAC object."""
        env = env()
        agent = DSAC(env, buffer_size=100, batch_size=10)
        while not agent.replay_buffer.full():
            agent.replay_buffer.push(transition)
        batch = agent.replay_buffer.sample_buffer(agent.batch_size)

        state = th.as_tensor(batch.obs, device=agent.device)
        action = th.as_tensor(batch.action, device=agent.device)
        tau = generate_quantiles(len(state), agent.num_quantiles, device=agent.device)

        z = agent.policy.z1(state, action, tau)
        z_target = agent.target_policy.z1(state, action, tau)

        # Assert error with the wrong shapes
        with pytest.raises(ValueError) as e_info:
            agent.huber_quantile_loss(z_target.flatten(), z, tau)
        assert "Target" in str(e_info.value)

        with pytest.raises(ValueError) as e_info:
            agent.huber_quantile_loss(z_target, z.flatten(), tau)
        assert "Predicted" in str(e_info.value)

        with pytest.raises(ValueError) as e_info:
            agent.huber_quantile_loss(z_target, z, tau.flatten())
        assert "Quantiles" in str(e_info.value)

        loss = agent.huber_quantile_loss(z_target, z, tau)

        assert isinstance(loss, th.Tensor)
        assert loss.shape == th.Size([])

    def test_update(self, env: Type[BaseEnv], transition):
        """Tests the update method of the DSAC object."""
        env = env()
        agent = DSAC(env, buffer_size=100, batch_size=10)
        while not agent.replay_buffer.full():
            agent.replay_buffer.push(transition)

        n_updates = 5
        agent._setup_learn(n_updates, "test")
        for _ in range(n_updates):
            agent.update()
        assert agent._n_updates == 5

    # def test_learn(self, env: Type[BaseEnv], transition):
    #     """Tests the learn method of the DSAC object."""
    #     env = env()
    #     learning_starts = 10
    #     agent = DSAC(
    #         env, buffer_size=100, batch_size=10, learning_starts=learning_starts
    #     )
    #
    #     num_steps = 15
    #     agent.learn(num_steps, "test")
    #     assert agent._n_updates == num_steps - learning_starts
