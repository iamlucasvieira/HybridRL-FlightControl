"""Module that implements the IDHP agent."""
from dataclasses import dataclass
from typing import List
from typing import Optional

import torch as th

from agents import BaseAgent
from agents.base_callback import ListCallback
from agents.idhp.incremental_model import IncrementalCitation
from agents.idhp.policy import IDHPPolicy
from envs import BaseEnv


class IDHP(BaseAgent):
    """Class that implements the IDHP algorithm."""

    def __init__(
        self,
        env: BaseEnv,
        discount_factor: float = 0.6,
        discount_factor_model: float = 0.8,
        verbose: int = 1,
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        seed: int = None,
        learning_rate: float = 0.08,
        actor_kwargs: Optional[dict] = None,
        critic_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize the IDHP algorithm.

        Args:
            policy: Policy to use.
            env: Environment to use.
            discount_factor (float): Discount factor.
            discount_factor_model (float): Discount factor for incremental model.
            verbose (int): Verbosity level.
            log_dir (str): Directory to save logs.
            save_dir (str): Directory to save models.
            seed (int): Seed for random number generator.
            beta_actor (float): Actor regularization parameter.
            learning_rate (float): Critic regularization parameter.
            hidden_layers (List[int]): Hidden layers for actor and critic.
        """
        # Make sure environment has the right observation and reward functions for IDHP
        env = self._setup_env(env)

        # Create the policy kwargs
        actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        critic_kwargs = {} if critic_kwargs is None else critic_kwargs
        default_policy_kwargs = {"learning_rate": learning_rate}
        actor_kwargs = actor_kwargs | default_policy_kwargs
        critic_kwargs = critic_kwargs | default_policy_kwargs
        policy_kwargs = {"actor_kwargs": actor_kwargs, "critic_kwargs": critic_kwargs}

        super().__init__(
            IDHPPolicy,
            env,
            verbose=verbose,
            log_dir=log_dir,
            save_dir=save_dir,
            seed=seed,
            policy_kwargs=policy_kwargs,
        )

        self.gamma = discount_factor

        # Initialize model
        self.model = IncrementalCitation(self.env, gamma=discount_factor_model)

        self.learning_data = IDHPLearningData(
            [0],
            [0],
        )

    def setup_model(self):
        """Setup model."""
        pass

    @staticmethod
    def _setup_env(env: BaseEnv) -> BaseEnv:
        """Adds the required reward and observation fucntion to env."""
        env.set_reward_function("sq_error")
        env.set_observation_function("states + ref")
        return env

    @property
    def actor(self) -> th.nn.Module:
        """Get actor."""
        return self.policy.actor

    @property
    def critic(self) -> th.nn.Module:
        """Get critic."""
        return self.policy.critic

    def _learn(
        self,
        total_steps: int,
        callback: ListCallback,
        log_interval: int,
        **kwargs,
    ):
        """Learn the policy."""
        obs_t, _ = self.env.reset()
        obs_t = th.tensor(obs_t, requires_grad=True, dtype=th.float32)

        while self.num_steps < total_steps:
            ###############################################
            # Act and collect state transition and reward #
            ###############################################

            # Increment timestep
            self.num_steps += 1

            # Sample and scale action
            action = self.actor(obs_t)
            critic_t = self.critic(obs_t)

            ##############
            # Get losses #
            ##############
            da_ds = []
            for i in range(action.shape[0]):
                grad_i = th.autograd.grad(
                    action[i],
                    obs_t,
                    grad_outputs=th.ones_like(action[i]),
                    retain_graph=True,
                )
                da_ds.append(grad_i[0])
            da_ds = th.stack(da_ds)

            with th.no_grad():
                # Step environment

                obs_t1, rew_t1, terminated, truncated, info = self.get_rollout(
                    action.detach().numpy(), obs_t.detach().numpy(), callback
                )

                done = terminated or truncated

                if done:
                    self.print(
                        f"Episode done with total steps {self.num_steps} '{info}'"
                    )
                    break
                callback.on_step()

                # Convert to tensors
                obs_t1 = th.tensor(obs_t1, requires_grad=True, dtype=th.float32)
                error_t1 = th.tensor(self.env.error[-1], dtype=th.float32)

                # Get the reward gradient with respect to the state at time t+1
                dr1_ds1 = -2 * error_t1 * self.env.tracked_state_mask

                critic_t1 = self.critic(obs_t1)

                # Get incremental model predictions of F and G at time t-1
                F_t_1 = th.tensor(self.model.F, dtype=th.float32)
                G_t_1 = th.tensor(self.model.G, dtype=th.float32)

                # Get loss gradients for actor and critic
                loss_gradient_a = self.actor.get_loss(
                    dr1_ds1, self.gamma, critic_t1, G_t_1
                )
                loss_gradient_c = self.critic.get_loss(
                    dr1_ds1, self.gamma, critic_t, critic_t1, F_t_1, G_t_1, da_ds
                )

            ########################
            # Update IDHP elements #
            ########################

            # Update actor network
            action = self.actor(obs_t, to_scale=False)
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
            self.model.update(self.env)

            ##########
            # Loging #
            ##########
            # Log to stable baseliens logger
            loss_gradient_a_mean = loss_gradient_a.mean().detach().numpy()
            loss_gradient_c_mean = loss_gradient_c.mean().detach().numpy()
            self.logger.record("train/actor_loss", loss_gradient_a_mean)
            self.logger.record("train/critic_loss", loss_gradient_c_mean)

            # Log to be used in wandb
            self.learning_data.loss_a.append(loss_gradient_a_mean.item())
            self.learning_data.loss_c.append(loss_gradient_c_mean.item())

            if log_interval is not None and self.num_steps % log_interval == 0:
                self.dump_logs()

            # Update observation
            obs_t = obs_t1  # Update obs

        return self


@dataclass
class IDHPLearningData:
    """Class that stores the data used for learning."""

    loss_a: List[float]
    loss_c: List[float]
