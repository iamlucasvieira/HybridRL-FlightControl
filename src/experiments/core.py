"""File that defines the core of the experiments."""
import os
import pathlib as pl
from typing import Optional

import matplotlib.pyplot as plt
import wandb
from rich.pretty import pprint
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from helpers.config import ConfigLinearAircraft
from helpers.tracking import TensorboardCallback
from helpers.misc import get_name
from helpers.paths import Path, set_wandb_path
from models.aircraft_environment import AircraftEnv


class Experiment:
    """Class that builds an experiment."""

    def __init__(self,
                 algorithm_name: str = "SAC",
                 env_name: str = "citation",
                 filename: str = "citation.yaml",
                 configuration: str = "symmetric",
                 task_name="aoa",
                 seed: Optional[int] = None,
                 dt: float = 0.1,
                 episode_steps: int = 100,
                 learning_steps: int = 1_000,
                 verbose: int = 2,
                 offline: bool = False,
                 project_name=""):
        """Initiates the experiment.

        args:
            algorithm_name: Name of the algorithm used.
            env_name: Environment to use.
            task_name: Task to perform.
            seed: Random seed.
            dt: Time step.
            episode_steps: Number of steps.
            learning_steps: Number of total learning steps.
            TO_TRAIN: Whether to train the model.
            TO_PLOT: Whether to plot the results.
            verbose: Verbosity level.
            name: Name of the experiment.
            tags: Tags of the experiment.
            offline: Whether to run offline.
            project_name: Name of the project.

        properties:
            config: Configuration of the experiment.
            verbose: Verbosity level.
            offline: Whether to run offline.
            env: Environment of the experiment.
            algo: Algorithm of the experiment.
            model: Model of the experiment.
            project_name: Name of the project.
            MODELS_PATH: Path to the models.
            LOGS_PATH: Path to the logs.
            wandb_run: Wandb run.
        """

        if offline:
            os.environ["WANDB_MODE"] = "offline"

        # Set the wandb log path
        set_wandb_path()

        self.config = ConfigLinearAircraft(
            algorithm=algorithm_name,
            env_name=env_name,
            filename=filename,
            configuration=configuration,
            seed=seed,
            dt=dt,
            episode_steps=episode_steps,
            learning_steps=learning_steps,
            task=task_name,
        )

        self.verbose = verbose
        self.offline = offline
        self.model = None
        self.wandb_run = None

        # Define project name and paths
        self.project_name = project_name if project_name else f"{env_name}-{algorithm_name}-{task_name}"
        self.MODELS_PATH = Path.models / self.project_name
        self.LOGS_PATH = Path.logs / self.project_name

        self.env = self.get_environment()
        self.algo = self.get_algorithm()

    def learn(self, name=None, tags=None, wandb_config={}, wandb_kwargs={}):
        """Learn the experiment."""
        if self.verbose > 0:
            pprint(self.config.asdict)

        # Get the environment and algorithm
        env = self.env
        algo = self.algo

        # Start wandb
        run = wandb.init(
            project=self.project_name,
            config=self.config.asdict | wandb_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,  # optional
            name=name,
            tags=tags,
            **wandb_kwargs,
        )

        self.wandb_run = run

        # If online, get the run name provided by wandb
        run_name = get_name([self.config.env_name, self.config.algorithm]) if self.offline else run.name

        # Create directories
        pl.Path.mkdir(self.MODELS_PATH / run_name, parents=True, exist_ok=True)
        pl.Path.mkdir(self.LOGS_PATH, parents=True, exist_ok=True)

        # Wrap environment with monitor
        env = Monitor(env, filename=f"{self.MODELS_PATH}/{run_name}")

        # Load wandb callback
        wandb_callback = WandbCallback(
            model_save_freq=100,
            # gradient_save_freq=config.episode_steps,
            model_save_path=f"{self.MODELS_PATH / run_name}",
            verbose=2)

        # Tensorboard callback
        tensorboard_callback = TensorboardCallback(verbose=2)

        # Create model
        model = algo("MlpPolicy",
                     env,
                     verbose=self.verbose,
                     tensorboard_log=self.LOGS_PATH,
                     seed=self.config.seed, )

        # Learn model
        model.learn(total_timesteps=self.config.learning_steps,
                    callback=[wandb_callback, tensorboard_callback],
                    log_interval=2,
                    tb_log_name=run_name,
                    progress_bar=True)

        # Replace previous latest-model with the new model
        model.save(f"{self.MODELS_PATH}/latest-model")

        self.model = model

    def get_environment(self):
        """Get environment."""
        if self.config.env_name == "citation":
            env = AircraftEnv(self.config)
        else:
            raise ValueError(f"Environment {self.config.env_name} not implemented.")

        return env

    def get_algorithm(self):
        """Get the algorithm."""
        if self.config.algorithm.lower() == "sac":
            algo = SAC
        else:
            raise ValueError(f"Algorithm {self.config.algorithm} not implemented.")
        return algo

    def load_model(self, model_name="olive-sun-4"):
        """Load a model file."""
        model = SAC.load(Path.models / self.project_name / model_name / "model.zip")
        self.model = model

    def run(self, n_times=1):
        """Run the experiment n times."""
        env = self.env

        for _ in range(n_times):
            obs = env.reset()

            for i in range(self.config.episode_steps):
                action, _states = self.model.predict(obs, deterministic=True)

                obs, reward, done, info = env.step(action)

                env.render()

                if wandb.run is not None:
                    wandb.log({f"reward": reward})
                    wandb.log({f"reference": env.reference[-1]})
                    wandb.log({f"state": env.track[-1]})

                if done:
                    print(f"finished at {i}")
                    break

        if wandb.run is not None:
            self.wandb_run.finish()

    def plot(self):
        """Plot a run of the experiment."""
        env = self.env
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(env.track)
        ax[0].plot(env.reference, '--')
        ax[1].plot(env.actions)
        plt.show()

    def finish_wandb(self):
        """Finish the wandb logging."""
        if wandb.run is not None:
            self.wandb_run.finish()


if __name__ == "__main__":
    main(TO_PLOT=True)
