"""Module for tracking training performance."""
import sys
import time

import numpy as np
import wandb

from agents import BaseCallback
from envs import BaseEnv, LTIEnv
from helpers.wandb_helpers import evaluate


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        time_elapsed = max(
            (time.time_ns() - self.agent.start_time) / 1e9, sys.float_info.epsilon
        )

        self.agent.logger.record("time/time_elapsed", time_elapsed)
        self.agent.logger.record("time/episodes", self.agent._episode_num)

        if self.agent.num_steps % self.agent.log_interval != 0 or wandb.run is None:
            return True

        if self.agent.rewards:
            wandb.log(
                {
                    "learning/rewards": self.agent.rewards[-1],
                    "learning/step": self.agent.num_steps,
                }
            )
        return True


class OnlineCallback(BaseCallback):
    """
    A custom callback to log the IDHP online learning data.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the agent after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.agent.num_steps % self.agent.log_interval != 0 or wandb.run is None:
            return True

        reference = self.env.reference[-2]
        state = self.env.track[-1]
        sq_error = self.env.sq_error[-1]
        action = self.env.actions[-1]
        step = self.agent.num_steps

        # Log Tracking performance
        for idx, state_name in enumerate(self.env.task.tracked_states):
            wandb.log(
                {
                    f"online/{state_name}_ref": reference[idx],
                    f"online/{state_name}": state[idx],
                    "online/step": step,
                }
            )
            wandb.log(
                {f"online/{state_name}_sq_error": sq_error[idx], "online/step": step}
            )

        for idx, action_name in enumerate(self.env.input_names):
            wandb.log(
                {f"online/action_{action_name}": action[idx], "online/step": step}
            )

        if isinstance(self.env, LTIEnv):
            # Log incremental model
            wandb.log(
                {
                    f"online/model_w_{idx}": i
                    for idx, i in enumerate(self.agent.model.theta.flatten())
                }
                | {"online/step": step}
            )
            wandb.log(
                {
                    f"online/model_error_{idx}": abs(i)
                    for idx, i in enumerate(self.agent.model.errors[-1].flatten())
                }
                | {"online/step": step}
            )

            # Log Actor
            actor_weights = (
                self.agent.actor.state_dict()["ff.0.weight"].flatten()[:3].numpy()
            )
            wandb.log(
                {
                    "online/actor_loss": self.agent.learning_data.loss_a[-1],
                    "online/step": step,
                }
            )
            wandb.log(
                {f"online/actor_w_{idx}": i for idx, i in enumerate(actor_weights)}
                | {"online/step": step}
            )

            # Log Critic
            critic_weights = (
                self.agent.critic.state_dict()["ff.0.weight"].flatten()[:3].numpy()
            )
            wandb.log(
                {
                    "online/critic_loss": self.agent.learning_data.loss_c[-1],
                    "online/step": step,
                }
            )
            wandb.log(
                {f"online/critic_w_{idx}": i for idx, i in enumerate(critic_weights)}
                | {"online/step": step}
            )
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        if wandb.run is not None:
            wandb.log({"online/mean_error": np.mean(self.env.sq_error)})
        return True


class IDHPCallback(BaseCallback):
    """Custom callback for IDHP."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Method called after each step."""
        if self.agent.num_steps % self.agent.log_interval != 0 or wandb.run is None:
            return True
        actor_lr = self.agent.actor.optimizer.param_groups[0]["lr"]
        critic_lr = self.agent.critic.optimizer.param_groups[0]["lr"]

        wandb.log({"idhp/actor_lr": actor_lr, "idhp/step": self.agent.num_steps})
        wandb.log({"idhp/critic_lr": critic_lr, "idhp/step": self.agent.num_steps})
        return True


class IDHPSACCallback(BaseCallback):
    """
    A custom callback to log the IDHP-SAC online learning data.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the agent after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.agent.num_steps % self.agent.log_interval != 0 or wandb.run is None:
            return True

        step = self.agent.num_steps

        # Log Actor
        actor_weights = (
            self.agent.actor.state_dict()["ff.2.weight"].flatten()[:3].numpy()
        )

        wandb.log(
            {f"actor/idhp_w_{idx}": i for idx, i in enumerate(actor_weights)}
            | {"train/step": step}
        )
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        if wandb.run is not None:
            wandb.log({"online/nmae": self.env.nmae})


class SACCallback(BaseCallback):
    """Callback for the SAC algorithm."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.best_nmae = np.inf

    def _on_step(self) -> bool:
        self.agent.logger.record("rollout/best_nmae", f"{self.best_nmae * 100 :.2f}%")
        return True

    def on_episode_end(self, episode_return) -> None:
        """Runs after each episode."""
        if not isinstance(self.env, BaseEnv):
            return
        self.agent.save(run=str(self.agent.num_steps))

        episode_length = self.env.current_time
        train_nmae = self.env.nmae
        eval_episode_return = evaluate(self.agent, self.env, to_wandb=False)
        eval_nmae = self.env.nmae
        eval_episode_length = self.env.current_time

        wandb.log(
            {
                "sac_train/episode_return": episode_return,
                "sac_train/steps": self.agent.num_steps,
            }
        )
        wandb.log(
            {
                "sac_train/episode_length": episode_length,
                "sac_train/steps": self.agent.num_steps,
            }
        )
        wandb.log(
            {"sac_train/nmae": train_nmae, "sac_train/steps": self.agent.num_steps}
        )

        wandb.log(
            {
                "sac_eval/episode_return": eval_episode_return,
                "sac_eval/steps": self.agent.num_steps,
            }
        )
        wandb.log(
            {
                "sac_eval/episode_length": eval_episode_length,
                "sac_eval/steps": self.agent.num_steps,
            }
        )
        wandb.log({"sac_eval/nmae": eval_nmae, "sac_eval/steps": self.agent.num_steps})

        if eval_nmae < self.best_nmae:
            self.best_nmae = eval_nmae
            self.agent.save(run="best")
            self.agent.logger.record(
                "rollout/best_nmae", f"{self.best_nmae * 100 :.2f}%"
            )


class ProgressCallback(BaseCallback):
    """Callback for the progressbar."""

    def __init__(self, verbose=0, progress=None):
        super().__init__(verbose)
        self.progress = progress

    def _on_training_start(self) -> None:
        """Method called before the training initialization."""
        self.training = self.progress.add_task(
            ":robot: Learning ", total=self.agent.total_steps
        )

    def _on_step(self) -> bool:
        """Method called after each step."""
        self.progress.update(self.training, advance=1)
        return True


AVAILABLE_CALLBACKS = {
    "tensorboard": TensorboardCallback,
    "online": OnlineCallback,
    "idhp_sac": IDHPSACCallback,
    "idhp": IDHPCallback,
    "sac": SACCallback,
    "progress": ProgressCallback,
}
