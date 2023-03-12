"""Module that tests the implementation of the SAC algorithm."""

import pytest

from agents.sac.sac import SAC
from agents.sac.policy import SACPolicy
from agents.sac.buffer import ReplayBuffer, Transition
from envs.citation.citation_env import CitationEnv
from envs.lti_citation.lti_env import LTIEnv
import torch as th
from copy import deepcopy

@pytest.fixture
def transition(env):
    """Create a Transition instance."""
    obs = env.reset()
    return Transition(obs=th.from_numpy(obs),
                      action=th.from_numpy(env.action_space.sample()),
                      reward=0,
                      obs_=th.from_numpy(obs),
                      done=False)


@pytest.mark.parametrize('env', [CitationEnv(), LTIEnv()], ids=['CitationEnv', 'LTIEnv'])
class TestSAC:
    """Test SAC class."""

    def test_init(self, env):
        """Test that SAC is correctly initialized."""
        sac = SAC(env)
        assert sac is not None

    def test_policy(self, env):
        """Test that SAC policy is correctly initialized."""
        sac = SAC(env)
        assert isinstance(sac.policy, SACPolicy)

    def test_target_policy(self, env):
        """Test that SAC target policy is correctly initialized and that it is a different object than policy."""
        sac = SAC(env)
        assert isinstance(sac.target_policy, SACPolicy)
        assert sac.target_policy is not sac.policy

    def test_replay_buffer(self, env):
        """Test that SAC replay buffer is correctly initialized."""
        sac = SAC(env)
        assert isinstance(sac.replay_buffer, ReplayBuffer)

    def test_get_critic_loss(self, env, transition):
        """Test that SAC critic loss is correctly computed."""
        sac = SAC(env)
        critic_loss = sac.get_critic_loss(transition)
        assert isinstance(critic_loss, th.Tensor)

    def test_get_actor_loss(self, env, transition):
        """Test that SAC actor loss is correctly computed."""
        sac = SAC(env)
        actor_loss = sac.get_actor_loss(transition)
        assert isinstance(actor_loss, th.Tensor)

    def test_update(self, env):
        """Test that SAC update is correctly computed."""
        sac = SAC(env,
                  buffer_size=10,
                  batch_size=5)
        obs = env.reset()
        while not sac.replay_buffer.full():
            action = env.action_space.sample()
            obs_tp1, reward, done, info = env.step(action)

            sac.replay_buffer.push(Transition(obs=obs,
                                              action=action,
                                              reward=reward,
                                              obs_=obs_tp1,
                                              done=done))
            obs = obs_tp1
        sac._setup_learn(total_timesteps=100)
        sac.update()
        # After one update gradients should exist
        assert sac.policy.actor.optimizer.param_groups[0]['params'][0].grad is not None
        assert sac.policy.critic_1.optimizer.param_groups[0]['params'][0].grad is not None
        assert sac.policy.critic_2.optimizer.param_groups[0]['params'][0].grad is not None

    @pytest.mark.parametrize('polyak, result', [(0, True), (0.5, False)], ids=['0', '0.5'])
    def test_update_target(self, env, polyak, result):
        """Test that SAC target update is correctly computed."""
        sac = SAC(env,
                  polyak=polyak)

        # Edit policy parameters
        for param in sac.policy.critic_1.parameters():
            param.data.copy_(th.ones_like(param.data))

        sac.update_target_networks()

        # If polyak is 0, target policy should be the same as policy
        for param, target_param in zip(sac.policy.critic_1.parameters(), sac.target_policy.critic_1.parameters()):
            assert th.allclose(param.data, target_param.data) == result

    @pytest.mark.parametrize('total_timesteps', [15, 100], ids=['15', '100'])
    def test_learn(self, env, total_timesteps):
        """Test that SAC learn is correctly computed."""
        sac = SAC(env,
                  buffer_size=10,
                  batch_size=5,
                  learning_starts=5)
        sac.learn(total_timesteps)
        assert sac.num_timesteps == total_timesteps
