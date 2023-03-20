"""SAC implemmenation using PyTorch."""
import os

import numpy as np
import torch as T
import torch.nn.functional as F
from buffer import ReplayBuffer
from networks import ActorNetwork
from networks import CriticNetwork
from networks import ValueNetwork


class Agent:
    """Agent class for SAC algorithm."""

    def __init__(
        self,
        alpha=0.0003,
        beta=0.0003,
        input_dims=[8],
        tau=0.005,
        env=None,
        gamma=0.99,
        n_actions=2,
        max_size=1000000,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
    ):
        """Initialize agent.
        args:
            alpha: Learning rate for actor network.
            beta: Learning rate for critic network.
            input_dims: Input dimensions.
            tau: Soft update parameter.
            env: Environment.
            gamma: Discount factor.
            n_actions: Number of actions.
            max_size: Maximum size of the replay buffer.
            layer1_size: Number of neurons in the first layer.
            layer2_size: Number of neurons in the second layer.
            batch_size: Batch size.
            reward_scale: Reward scaling factor.
        """
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.reward_scale = reward_scale

        # Define networks
        self.actor = ActorNetwork(
            alpha, input_dims, n_actions=n_actions, max_action=env.action_space.high
        )
        self.critic_1 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_1"
        )
        self.critic_2 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_2"
        )
        self.value = ValueNetwork(beta, input_dims)
        self.target_value = ValueNetwork(beta, input_dims, name="target_value")

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        """Choose action based on observation.
        args:
            observation: Observation.
        returns:
            action: Action.
        """
        state = T.Tensor(np.array([observation])).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparamterize=False)

        # Remove from device to cpu, detach, and convert to numpy array
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        """Store experience in replay buffer.
        args:
            state: State.
            action: Action.
            reward: Reward.
            new_state: New state.
            done: Done.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        # At the start target_value and value are the same, this allows us to pass different tau at start
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone()
                + (1 - tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        """Save models."""
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        """Load models."""
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

    def learn(self):
        """Learn from experience."""
        # Check if there are enough samples in the replay buffer
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)  # Using the target network
        value_[done] = 0.0

        def get_critic_value(_state, _reparamterize=False):
            """Get critic value.
            args:
                state: State.
            returns:
                critic_value: Critic value.
            """
            actions, _log_probs = self.actor.sample_normal(
                _state, reparamterize=_reparamterize
            )
            _log_probs = _log_probs.view(-1)
            q1_new_policy = self.critic_1(_state, actions)
            q2_new_policy = self.critic_2(_state, actions)
            _critic_value = T.min(q1_new_policy, q2_new_policy)
            _critic_value = _critic_value.view(-1)

            return _critic_value, _log_probs

        # Update value network
        critic_value, log_probs = get_critic_value(state)
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Update actor network
        critic_value, log_probs = get_critic_value(state, _reparamterize=True)
        actor_loss = T.mean(log_probs - critic_value)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Update critic networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.reward_scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1(state, action).view(-1)
        q2_old_policy = self.critic_2(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
