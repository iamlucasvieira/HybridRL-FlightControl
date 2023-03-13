"""Module that implemment IDHP agent."""
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback
from envs import BaseEnv
from agents.idhp.policy import IDHPPolicy
from agents.idhp.incremental_model import IncrementalCitation
from dataclasses import dataclass
from helpers.callbacks import OnlineCallback
from typing import Type, Tuple, List
import numpy as np
import wandb


class IDHP(BaseAlgorithm):
    """Class that implements the IDHP algorithm."""

    policy_aliases = {'default': IDHPPolicy}

    def __init__(self,
                 policy: str,
                 env: Type[BaseEnv],
                 discount_factor: float = 0.6,
                 discount_factor_model: float = 0.8,
                 verbose: int = 1,
                 tensorboard_log: str = None,
                 seed: int = None,
                 learning_rate: float = 0.08,
                 hidden_size: int = 10,
                 **kwargs):
        """Initialize the IDHP algorithm.

        Args:
            policy: Policy to use.
            env: Environment to use.
            discount_factor (float): Discount factor.
            discount_factor_model (float): Discount factor for incremental model.
            verbose (int): Verbosity level.
            tensorboard_log (str): Path to tensorboard log.
            seed (int): Seed for random number generator.
            beta_actor (float): Actor regularization parameter.
            learning_rate (float): Critic regularization parameter.
            hidden_size (int): Hidden size for actor and critic networks.
        """
        # Make sure environment has the right observation and reward functions for IDHP
        env = self._setup_env(env)

        super(IDHP, self).__init__(policy,
                                   env,
                                   learning_rate,
                                   verbose=verbose,
                                   tensorboard_log=tensorboard_log,
                                   seed=seed)

        self.gamma = discount_factor

        # Policy kwargs
        self.policy_kwargs = {
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
        }

        # Initialize model kwargs
        self.model_kwargs = {
            "gamma": discount_factor_model,
        }

        self.policy, self.actor, self.critic = None, None, None

        self._setup_model()

        self.learning_data = IDHPLearningData([0],
                                              [0], )

        self.log_interval = None

    @staticmethod
    def _setup_env(env: Type[BaseEnv]) -> Type[BaseEnv]:
        """Adds the required reward and observation fucntion to env."""
        env.set_reward_function('sq_error')
        env.set_observation_function('states + ref')

        return env

    def _setup_model(self) -> None:
        """Setup the model."""
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(self.observation_space,
                                        self.action_space,
                                        **self.policy_kwargs)

        self.model = IncrementalCitation(self._env, **self.model_kwargs)
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
        self.log_interval = log_interval
        online_callback = OnlineCallback()
        callback = online_callback if callback is None else callback + [online_callback]

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar, )

        # Because IDHP is online, the episode step is equal to the learning step
        self._env.env.episode_steps = total_timesteps
        self._env.env.episode_length = total_timesteps * self._env.dt

        callback.on_training_start(locals(), globals())
        obs_t = th.tensor(np.array([self._env.reset()]), requires_grad=True, dtype=th.float32)

        while self.num_timesteps < total_timesteps:
            ###############################################
            # Act and collect state transition and reward #
            ###############################################

            # Increment timestep
            self.num_timesteps += 1

            # Sample and scale action
            action = scale_action(self.actor(obs_t), self._env.action_space)
            critic_t = self.critic(obs_t)

            ##############
            # Get losses #
            ##############
            da_ds = []
            for i in range(action.shape[1]):
                grad_i = th.autograd.grad(action[:, i], obs_t, grad_outputs=th.ones_like(action[:, i]), retain_graph=True)
                da_ds.append(grad_i[0])
            da_ds = th.cat(da_ds)

            with th.no_grad():
                # Step environment
                obs_t1, rew_t1, done, info = self.env.step(action.detach().numpy())

                if done:
                    print('Episode done', self.num_timesteps)
                    break
                callback.on_step()

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(info, done)

                # Convert to tensors
                obs_t1 = th.tensor(obs_t1, requires_grad=True, dtype=th.float32)
                error_t1 = th.tensor(self._env.error[-1], dtype=th.float32)

                # Get the reward gradient with respect to the state at time t+1
                dr1_ds1 = - 2 * error_t1 * self._env.tracked_state_mask

                critic_t1 = self.critic(obs_t1)

                # Get incremental model predictions of F and G at time t-1
                F_t_1 = th.tensor(self.model.F, dtype=th.float32)
                G_t_1 = th.tensor(self.model.G, dtype=th.float32)

                # Get loss gradients for actor and critic
                loss_gradient_a = self.actor.get_loss(dr1_ds1, self.gamma, critic_t1, G_t_1)
                loss_gradient_c = self.critic.get_loss(dr1_ds1, self.gamma, critic_t, critic_t1, F_t_1, G_t_1, da_ds)

            ########################
            # Update IDHP elements #
            ########################

            # Update actor network
            action = scale_action(self.actor(obs_t), self._env.action_space)  # Need to resample due to previous detach
            self.actor.optimizer.zero_grad()
            loss_a = action * loss_gradient_a
            loss_a.backward(gradient=th.ones_like(loss_gradient_a))
            self.actor.optimizer.step()

            # Update critic network
            self.critic.optimizer.zero_grad()
            loss_c = critic_t * loss_gradient_c
            loss_c.backward(gradient=th.ones_like(loss_gradient_c))
            self.critic.optimizer.step()

            # Update incremental model
            self.model.update(self._env)

            ##########
            # Loging #
            ##########
            # Log to stable baseliens logger
            loss_gradient_a_mean = loss_gradient_a.mean().detach().numpy()
            loss_gradient_c_mean = loss_gradient_c.mean().detach().numpy()
            self.logger.record("train/actor_loss", loss_gradient_a_mean)
            self.logger.record("train/critic_loss", loss_gradient_c_mean)
            self.logger.record("train/steps", self.num_timesteps)

            # Log to be used in wandb
            self.learning_data.loss_a.append(loss_gradient_a_mean.item())
            self.learning_data.loss_c.append(loss_gradient_c_mean.item())

            if log_interval is not None and self.num_timesteps % log_interval == 0:
                self.logger.dump()

            # Update observation
            obs_t = obs_t1  # Update obs

        callback.on_training_end()

        return self

    @property
    def _env(self):
        """Return the environment from within the wrapper."""
        return self.env.envs[0]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["policy.actor", "policy.critic"]

        return state_dicts, []


def scale_action(action: np.ndarray, action_space) -> np.ndarray:
    """Scale the action to the correct range."""
    low, high = action_space.low[0], action_space.high[0]
    return action * (high - low) / 2 + (high + low) / 2


@dataclass
class IDHPLearningData:
    """Class that stores the data used for learning."""
    loss_a: List[float]
    loss_c: List[float]
