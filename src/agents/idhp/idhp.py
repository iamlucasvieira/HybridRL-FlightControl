"""Module that implements the IDHP agent."""
from dataclasses import dataclass
from typing import List, Optional, Type

import numpy as np
import torch as th

from agents import BaseAgent
from agents.base_callback import ListCallback
from agents.idhp.excitation import get_excitation_function
from agents.idhp.incremental_model import IncrementalCitation
from agents.idhp.policy import IDHPPolicy
from envs import BaseEnv, CitationEnv
from envs.observations import get_observation
from helpers.torch_helpers import to_tensor


class IDHP(BaseAgent):
    """Class that implements the IDHP algorithm."""

    name = "IDHP"

    def __init__(
        self,
        env: Type[BaseEnv],
        discount_factor: float = 0.6,
        discount_factor_model: float = 0.8,
        verbose: int = 1,
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        seed: int = None,
        actor_kwargs: Optional[dict] = None,
        critic_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
        excitation: Optional[str] = None,
        lr_a_low: float = 0.005,
        lr_a_high: float = 0.08,
        lr_c_low: float = 0.0005,
        lr_c_high: float = 0.005,
        lr_threshold: float = 0.01,
        t_warmup: int = 100,
        actor_observation_type: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the IDHP algorithm.

        Args:
            env (Type[BaseEnv]): Environment to use.
            discount_factor (float, optional): Discount factor for the reward. Defaults to 0.6.
            discount_factor_model (float, optional): Discount factor for the incremental model. Defaults to 0.8.
            verbose (int, optional): Verbosity level. Defaults to 1.
            log_dir (Optional[str], optional): Directory to save logs. Defaults to None.
            save_dir (Optional[str], optional): Directory to save models. Defaults to None.
            seed (int, optional): Seed for the random number generator. Defaults to None.
            actor_kwargs (Optional[dict], optional): Keyword arguments for the actor. Defaults to None.
            critic_kwargs (Optional[dict], optional): Keyword arguments for the critic. Defaults to None.
            device (Optional[str], optional): Device to use. Defaults to None.
            excitation (Optional[str], optional): Excitation function to use. Defaults to None.
            lr_a_low (float, optional): Lower bound for the learning rate of the actor. Defaults to 0.005.
            lr_a_high (float, optional): Upper bound for the learning rate of the actor. Defaults to 0.08.
            lr_c_low (float, optional): Lower bound for the learning rate of the critic. Defaults to 0.0005.
            lr_c_high (float, optional): Upper bound for the learning rate of the critic. Defaults to 0.005.
            lr_threshold (float, optional): Threshold for the learning rate. Defaults to 1 [deg].
            t_warmup (int, optional): Number of warmup steps before allowing adaptive learning rate. Defaults to 100.
        """
        # Make sure environment has the right observation and reward functions for IDHP
        env = self._setup_env(env)
        n_states = env.n_states
        if isinstance(env, CitationEnv):
            # Removes the final three states from the Citation model
            self.states = ["p", "q", "r", "alpha", "theta", "phi", "beta"]
            states_idx = [env.states_name.index(s) for s in self.states]
            mask = np.zeros(n_states, dtype=bool)
            mask[states_idx] = True
            self.states_mask = mask
        else:
            self.states = env.states_name
            self.states_mask = np.array([True] * n_states)

        self.tracked_states = np.zeros((env.task.mask.sum(), len(self.states)))

        for i, state in enumerate(env.task.tracked_states):
            self.tracked_states[i][self.states.index(state)] = 1

        # Create the policy kwargs
        actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        critic_kwargs = {} if critic_kwargs is None else critic_kwargs

        actor_kwargs = {
            "lr_low": lr_a_low,
            "lr_high": lr_a_high,
            "lr_threshold": lr_threshold,
            **actor_kwargs,
        }
        critic_kwargs = {
            "lr_low": lr_c_low,
            "lr_high": lr_c_high,
            "lr_threshold": lr_threshold,
            "n_states": np.sum(self.states_mask),
            **critic_kwargs,
        }
        policy_kwargs = {"actor_kwargs": actor_kwargs, "critic_kwargs": critic_kwargs}

        super().__init__(
            IDHPPolicy,
            env,
            verbose=verbose,
            log_dir=log_dir,
            save_dir=save_dir,
            seed=seed,
            policy_kwargs=policy_kwargs,
            device=device,
        )

        self.gamma = discount_factor
        self.t_warmup = t_warmup

        # Initialize model
        self.model = IncrementalCitation(
            self.env, self.states_mask, gamma=discount_factor_model
        )

        self.learning_data = IDHPLearningData(
            [0],
            [0],
        )

        self.excitation_function = (
            None if excitation is None else get_excitation_function(excitation)
        )

        self.actor_observation = (
            None
            if actor_observation_type is None
            else get_observation(actor_observation_type)
        )

    def setup_model(self):
        """Setup model."""
        pass

    @staticmethod
    def _setup_env(env: Type[BaseEnv]) -> Type[BaseEnv]:
        """Adds the required reward and observation fucntion to env."""
        env.set_reward_function("sq_error")
        if isinstance(env, CitationEnv):
            env.set_observation_function("idhp_citation")

        else:
            env.set_observation_function("states + error")
        return env

    @property
    def actor(self) -> th.nn.Module:
        """Get actor."""
        return self.policy.actor

    @property
    def critic(self) -> th.nn.Module:
        """Get critic."""
        return self.policy.critic

    def get_observation_actor(self, obs):
        if self.actor_observation is not None:
            obs_t_actor = to_tensor(
                self.actor_observation(self.env), device=self.device
            )
        else:
            obs_t_actor = obs

        return obs_t_actor

    def _learn(
        self,
        total_steps: int,
        callback: ListCallback,
        log_interval: int,
        **kwargs,
    ):
        """Learn the policy."""
        obs_t, _ = self.env.reset()
        obs_t = th.tensor(
            obs_t, requires_grad=True, dtype=th.float32, device=self.device
        )

        while self.num_steps < total_steps:
            ###############################################
            # Act and collect state transition and reward #
            ###############################################

            # Increment timestep
            self.num_steps += 1

            # Sample and scale action
            with th.no_grad():
                actor_obs = self.get_observation_actor(obs_t)
                actor_obs.requires_grad = True
            action = self.actor(actor_obs)
            critic_t = self.critic(obs_t)

            ##############
            # Get losses #
            ##############
            da_ds = []
            for i in range(action.shape[0]):
                grad_i = th.autograd.grad(
                    action[i],
                    actor_obs,
                    grad_outputs=th.ones_like(action[i]),
                    retain_graph=True,
                )
                da_ds.append(grad_i[0])
            da_ds = th.stack(da_ds)
            # Only keeps the gradients with respect to the states and not the errors
            da_ds = da_ds[:, : -self.env.error[-1].shape[0]]

            da_ds_all_states = th.zeros((da_ds.shape[0], len(self.states)))
            da_ds_all_states[: da_ds.shape[0], : da_ds.shape[1]] = da_ds
            with th.no_grad():
                # Step environment

                if self.excitation_function is not None:
                    action += self.excitation_function(self.env)

                obs_t1, rew_t1, terminated, truncated, info = self.get_rollout(
                    action.cpu().detach().numpy(),
                    obs_t.cpu().detach().numpy(),
                    callback,
                )

                done = terminated or truncated

                if done:
                    self.print(
                        f"Episode done with total steps {self.num_steps} '{info}'"
                    )
                    break
                callback.on_step()

                # Convert to tensors
                obs_t1 = th.tensor(
                    obs_t1, requires_grad=True, dtype=th.float32, device=self.device
                )
                error_t1 = th.tensor(
                    self.env.error[-1], dtype=th.float32, device=self.device
                )

                # Get the reward gradient with respect to the state at time t+1
                dr1_ds1 = (
                    -2
                    * error_t1
                    @ th.as_tensor(
                        self.tracked_states, dtype=th.float32, device=self.device
                    )
                )

                critic_t1 = self.critic(obs_t1)

                # Get incremental model predictions of F and G at time t-1
                F_t_1 = th.tensor(self.model.F, dtype=th.float32, device=self.device)
                G_t_1 = th.tensor(self.model.G, dtype=th.float32, device=self.device)

                # Get loss gradients for actor and critic
                loss_gradient_a = self.actor.get_loss(
                    dr1_ds1, self.gamma, critic_t1, G_t_1
                )
                loss_gradient_c = self.critic.get_loss(
                    dr1_ds1,
                    self.gamma,
                    critic_t,
                    critic_t1,
                    F_t_1,
                    G_t_1,
                    da_ds_all_states,
                )

            ########################
            # Update IDHP elements #
            ########################

            # Update actor network
            action = self.actor(self.get_observation_actor(obs_t))
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

            ########################
            # Update learning rate #
            ########################
            if self.num_steps >= self.t_warmup:
                n_steps_average = 50
                if self.num_steps % n_steps_average == 0:
                    sq_error = np.array(self.env.sq_error).flatten()
                    mse = np.sqrt(sq_error[-n_steps_average:]).mean()

                    self.critic.update_learning_rate(mse)
                    self.actor.update_learning_rate(mse)

            ##########
            # Loging #
            ##########
            # Log to stable baseliens logger
            loss_gradient_a_mean = loss_gradient_a.mean().cpu().detach().numpy()
            loss_gradient_c_mean = loss_gradient_c.mean().cpu().detach().numpy()
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
