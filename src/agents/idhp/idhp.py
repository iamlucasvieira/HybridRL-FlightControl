"""Module that implemment IDHP agent."""
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback
from envs.lti_citation.aircraft_environment import AircraftEnv
from agents.idhp.policy import IDHPPolicy
from agents.idhp.incremental_model import IncrementalLTIAircraft
from agents.idhp.idhp_data import IDHPLearningData
from typing import Type, Optional, Dict, Any
import numpy as np


class IDHP(BaseAlgorithm):
    """Class that implements the IDHP algorithm."""

    policy_aliases = {'default': IDHPPolicy}

    def __init__(self,
                 policy: str,
                 env: Type[AircraftEnv],
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 1,
                 tensorboard_log: str = None,
                 seed: int = None,
                 learning_starts: int = 0,
                 **kwargs):
        """Initialize the IDHP algorithm.

        Args:
            gamma (float): Discount factor.
            learning_starts: how many steps of the model to collect transitions for before learning starts
        """
        # Make sure environment has the right observation and reward functions for IDHP
        env = self._setup_env(env)

        super(IDHP, self).__init__(policy,
                                   env,
                                   learning_rate,
                                   verbose=verbose,
                                   tensorboard_log=tensorboard_log,
                                   policy_kwargs=policy_kwargs,
                                   seed=seed)

        self.gamma = gamma

        self.policy, self.actor, self.critic = None, None, None

        self.learning_starts = learning_starts
        self.learning_data = []

        self._setup_model()

    @staticmethod
    def _setup_env(env: Type[AircraftEnv]) -> Type[AircraftEnv]:
        """Adds the required reward and observation fucntion to env."""
        env.set_reward_function('error')
        env.set_observation_function('states + ref')
        return env

    def _setup_model(self) -> None:
        """Setup the model."""
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(self.observation_space,
                                        self.action_space,
                                        **self.policy_kwargs)

        self.model = IncrementalLTIAircraft(self._env)
        self.actor = self.policy.actor
        self.critic = self.policy.critic

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        obs_t = torch.tensor(np.array([self._env.reset()]), requires_grad=True, dtype=torch.float32)

        while self.num_timesteps < total_timesteps:
            ###############################################
            # Act and collect state transition and reward #
            ###############################################

            # Increment timestep
            self.num_timesteps += 1

            # Sample action
            action = self.actor(obs_t)

            # Backpropagate action gradient
            action.backward()
            obs_grad = obs_t.grad
            self.actor.zero_grad()

            # Step environment
            obs_t1, rew_t1, done, info = self.env.step(action.detach().numpy())

            # Store data
            self.learning_data.append(IDHPLearningData(action, obs_t1, rew_t1, done))
            self.model.increment(self._env, action.detach().numpy())

            # Convert to tensors
            obs_t1 = torch.tensor(obs_t1, requires_grad=True, dtype=torch.float32)
            rew_t1 = torch.tensor(rew_t1, dtype=torch.float32)

            if done:
                print('Episode done', self.num_timesteps)
                break
                # raise NotImplementedError

            ###############
            # Critic loss #
            ###############

            # Get the reward at time t+1 and build the cost function
            dr_ds = 2 * rew_t1

            # Get the critic at time t and t+1
            critic_t = self.critic(obs_t)
            critic_t1 = self.critic(obs_t1)

            # Get incremental model predictions of F and G at time t-1
            F_t_1 = torch.tensor(self.model.F, dtype=torch.float32)
            G_t_1 = torch.tensor(self.model.G, dtype=torch.float32)

            # Critic error

            e_c = -(dr_ds * self._env.tracked_state_mask + self.gamma * critic_t1) @ (
                    F_t_1 + G_t_1 @ obs_grad[:, :-1]) + critic_t

            ##############
            # Actor loss #
            ##############
            e_a = -(dr_ds + self.gamma * critic_t1) @ G_t_1

            ###################
            # Update networks #
            ###################

            # Backpropagate critic error
            self.critic.optimizer.zero_grad()
            e_c.backward(gradient=torch.ones_like(e_c * -1), retain_graph=True)

            # Backpropagate actor error
            self.actor.optimizer.zero_grad()
            e_a.backward(gradient=torch.ones_like(e_a * -1))

            # Update networks
            self.critic.optimizer.step()
            self.actor.optimizer.step()

            # Update incremental model
            self.model.update(self._env)

            # Update observation
            obs_t = obs_t1  # Update obs

        # callback.on_training_end()

        return self

    def _sample_action(self):
        return [0.1]  # self.actor(self.observation_space.sample())

    def collect_rollouts(self, callback: MaybeCallback = None):
        """Collects the experience with the environment."""

        callback.on_rollout_start()

        action = self._sample_action()

        new_obs, reward, done, info = self.env.step(action)

        self.num_timesteps += 1
        self.learning_data.append(IDHPLearningData(action, new_obs, reward, done))

        callback.update_locals(locals())

        return new_obs, reward, done, info, action

    @property
    def _env(self):
        """Return the environment from within the wrapper."""
        return self.env.envs[0]
