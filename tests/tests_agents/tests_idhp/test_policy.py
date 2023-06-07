"""Module that tests the IDHP policy."""
import numpy as np
import pytest
from torch import nn

from agents.idhp.policy import Actor, Critic, IDHPPolicy
from envs import CitationEnv, LTIEnv


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestIDHPPolicy:
    """Class that tests the IDHP policy."""

    @pytest.fixture
    def policy(self, env):
        env = env()
        obseration_space = env.observation_space
        action_space = env.action_space
        policy = IDHPPolicy(obseration_space, action_space)
        return policy

    def test_init(self, env, policy):
        """Test the IDHP policy."""
        assert policy is not None
        assert isinstance(policy.actor, Actor)
        assert isinstance(policy.critic, Critic)

    def test_predict(self, env, policy):
        """Test the IDHP policy."""
        env = env()
        obs, _ = env.reset()
        action = policy.predict(obs)
        assert action.shape == env.action_space.shape

    def test_actor_kwargs(self, env):
        """Test the IDHP policy."""
        env = env()
        policy = IDHPPolicy(
            env.observation_space,
            env.action_space,
            actor_kwargs={"hidden_layers": [10, 15, 20]},
        )
        assert policy.actor.num_hidden_layers == 3

    def test_critic_kwargs(self, env):
        """Test the IDHP policy."""
        env = env()
        policy = IDHPPolicy(
            env.observation_space,
            env.action_space,
            critic_kwargs={"hidden_layers": [15]},
        )
        assert policy.critic.num_hidden_layers == 1

    def test_adaptive_lr(self, env):
        """Test the IDHP policy."""
        env = env()
        lr_low = 0
        lr_high = 1
        lr_threshold = 0.5
        policy = IDHPPolicy(
            env.observation_space,
            env.action_space,
            critic_kwargs={
                "lr_low": lr_low,
                "lr_high": lr_high,
                "lr_threshold": lr_threshold,
            },
            actor_kwargs={
                "lr_low": lr_low,
                "lr_high": lr_high,
                "lr_threshold": lr_threshold,
            },
        )
        # Assert that the learning rates are high at start
        assert policy.actor.optimizer.param_groups[0]["lr"] == lr_high
        assert policy.critic.optimizer.param_groups[0]["lr"] == lr_high

        # Test if rates become low when loss is low
        loss_low = 0.1
        policy.actor.update_learning_rate(loss_low)
        policy.critic.update_learning_rate(loss_low)

        assert policy.actor.optimizer.param_groups[0]["lr"] == lr_low
        assert policy.critic.optimizer.param_groups[0]["lr"] == lr_low

        # Test if rates become high when loss is high
        loss_high = 0.6
        policy.actor.update_learning_rate(loss_high)
        policy.critic.update_learning_rate(loss_high)

        assert policy.actor.optimizer.param_groups[0]["lr"] == lr_high
        assert policy.critic.optimizer.param_groups[0]["lr"] == lr_high


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestActor:
    """Class that tests the actor network."""

    def test_init(self, env):
        """Test the actor network."""
        env = env()
        actor = Actor(env.observation_space, env.action_space)
        assert actor is not None
        assert isinstance(actor.ff, nn.Sequential)

    def test_forward(self, env):
        """Test the actor network."""
        env = env()
        actor = Actor(env.observation_space, env.action_space)
        obs, _ = env.reset()
        action = actor(obs)
        assert action.shape == env.action_space.shape

    def test_get_loss(self, env):
        """Test the actor network."""
        env = env()
        n_actions = env.action_space.shape[0]
        n_states = (
            env.observation_space.shape[0] - 1
        )  # Minus 1 to not include the reference
        actor = Actor(env.observation_space, env.action_space)
        dr1_ds1 = np.random.rand(1, n_states)
        gamma = 1
        critic_t1 = np.random.rand(1, n_states)
        G_t_1 = np.random.rand(n_states, n_actions)
        shape_actor_output = (1, n_actions)
        loss = actor.get_loss(dr1_ds1, gamma, critic_t1, G_t_1)
        assert loss.shape == shape_actor_output


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestCritic:
    """Class that tests the critic network."""

    def test_init(self, env):
        """Test the critic network."""
        env = env()
        critic = Critic(env.observation_space, env.action_space)
        assert critic is not None
        assert isinstance(critic.ff, nn.Sequential)

    def test_forward(self, env):
        """Test the critic network forward pass."""
        env = env()
        critic = Critic(
            env.observation_space,
            env.action_space,
            n_states=env.observation_space.shape[0] - 1,
        )
        obs, _ = env.reset()
        value = critic(obs)
        assert value.shape == (env.observation_space.shape[0] - 1,)

    # def test_get_loss(self, env):
    #     """Test the critic network loss."""
    #     env = env()
    #     critic = Critic(env.observation_space, env.action_space)
    #
    #     # The aircraft states does not include the reference state. That is why the -1
    #     n_actions = env.action_space.shape[0]
    #     n_states = (
    #         env.observation_space.shape[0] - 1
    #     )  # Minus 1 to not include the reference
    #     dr1_ds1 = np.random.rand(1, n_states)
    #     gamma = 1
    #     critic_t = np.random.rand(1, n_states)
    #     critic_t1 = np.random.rand(1, n_states)
    #     F_t_1 = np.random.rand(n_states, n_states)
    #     G_t_1 = np.random.rand(n_states, n_actions)
    #     # The obs grad includes the reference state. That is why the +1
    #     obs_grad = np.random.rand(n_actions, n_states + 1)
    #
    #     loss = critic.get_loss(
    #         dr1_ds1, gamma, critic_t, critic_t1, F_t_1, G_t_1, obs_grad
    #     )
    #     shape_critic_output = (1, n_states)
    #     assert loss.shape == shape_critic_output
