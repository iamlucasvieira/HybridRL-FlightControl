"""Module for tracking training performance."""
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np
import os
import time
import sys


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)

    Retrieved from: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    # def _init_callback(self) -> None:
    #     # Create folder if needed
    #     if self.save_path is not None:
    #         os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        time_elapsed = max((time.time_ns() - self.model.start_time) / 1e9, sys.float_info.epsilon)

        self.logger.record("time/time_elapsed", time_elapsed)
        self.logger.record("time/episodes", self.model._episode_num)

        return True


class OnlineCallback(BaseCallback):
    """
    A custom callback to log the IDHP online learning data.
    """

    def __init__(self, verbose=0):
        super(OnlineCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        reference = self.model._env.reference[-2]
        state = self.model._env.track[-1]
        sq_error = self.model._env.sq_error[-1]
        step = self.model.num_timesteps

        # Log Tracking performance
        wandb.log({"online/reference": reference,
                   "online/state": state,
                   "train/step": step})
        wandb.log({"online/sq_error": sq_error,
                   "train/step": step})

        # Log incremental model
        wandb.log(
            {f"model/w_{idx}": i for idx, i in enumerate(self.model.model.theta.flatten())} | {"train/step": step})
        wandb.log({f"model/error_{idx}": abs(i) for idx, i in enumerate(self.model.model.errors[-1].flatten())} | {
            "train/step": step})

        # Log Actor
        actor_weights = self.model.actor.state_dict()['ff.0.weight'].flatten()[:3].numpy()
        wandb.log({f"actor/loss": self.model.learning_data.loss_a[-1], "train/step": step})
        wandb.log({f"actor/w_{idx}": i for idx, i in enumerate(actor_weights)} | {"train/step": step})

        # Log Critic
        critic_weights = self.model.critic.state_dict()['ff.0.weight'].flatten()[:3].numpy()
        wandb.log({f"critic/loss": self.model.learning_data.loss_c[-1], "train/step": step})
        wandb.log({f"critic/w_{idx}": i for idx, i in enumerate(critic_weights)} | {"train/step": step})
        return True
