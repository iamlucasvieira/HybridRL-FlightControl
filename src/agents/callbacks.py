"""Module for tracking training performance."""
import sys
import time

import numpy as np
import wandb

from agents import BaseCallback


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
        wandb.log(
            {"online/reference": reference, "online/state": state, "online/step": step}
        )
        wandb.log({"online/sq_error": sq_error, "online/step": step})
        wandb.log({"online/action": action, "online/step": step})

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


AVAILABLE_CALLBACKS = {
    "tensorboard": TensorboardCallback,
    "online": OnlineCallback,
    "idhp_sac": IDHPSACCallback,
    "idhp": IDHPCallback,
}
