"""File that defines the core of the experiments."""
import os
import pathlib as pl
from typing import Optional
import yaml
import itertools

import matplotlib.pyplot as plt
import wandb
from rich.pretty import pprint
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from helpers.config import ConfigLinearAircraft, ConfigExperiment
from helpers.tracking import TensorboardCallback
from helpers.misc import get_name
from helpers.paths import Path, set_wandb_path
from models.aircraft_environment import AircraftEnv
from agents.seres_dsac import DSAC


class Sweep:
    """Class that builds an experiment."""

    def __init__(self,
                 config: Optional[ConfigLinearAircraft] = None,
                 algorithm_name: str = "SAC",
                 env_name: str = "citation",
                 filename: str = "citation.yaml",
                 configuration: str = "symmetric",
                 task_name="q_sin",
                 seed: Optional[int] = None,
                 dt: float = 0.1,
                 episode_steps: int = 100,
                 learning_steps: int = 1_000,
                 verbose: int = 2,
                 offline: bool = False,
                 project_name="",
                 log_interval: int = 1,
                 reward_type: str = "sq_error",
                 observation_type: str = "error",
                 evaluate: int = 1, ):
        """Initiates the experiment.

        args:
            config: configuration of the experiment
            algorithm_name: Name of the algorithm used.
            env_name: Environment to use.
            filename: Name of the file to load the aircraft configuration from.
            task_name: Task to perform.
            seed: Random seed.
            dt: Time step.
            episode_steps: Number of steps.
            learning_steps: Number of total learning steps.
            verbose: Verbosity level.
            offline: Whether to run offline.
            project_name: Name of the project.
            log_interval: Interval to log.
            reward_type: Type of reward.
            observation_type: Type of observation.
            evaluate: Number of times to run the environment after learning.

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

        if config is None:
            self.config = ConfigLinearAircraft(
                algorithm=algorithm_name.upper(),
                env_name=env_name,
                filename=filename,
                configuration=configuration,
                seed=seed,
                dt=dt,
                episode_steps=episode_steps,
                learning_steps=learning_steps,
                task=task_name,
                reward_type=reward_type,
                observation_type=observation_type,
                evaluat=evaluate,
                log_interval=log_interval,
            )
        else:
            self.config = config

        self.verbose = verbose
        self.offline = offline
        self.model = None
        self.wandb_run = None

        # Define project name and paths
        self.project_name = project_name if project_name else f"{self.config.env_name}-{self.config.algorithm}-{self.config.task}"
        self.MODELS_PATH = Path.models / self.project_name
        self.LOGS_PATH = Path.logs / self.project_name

        self.env = self.get_environment()
        self.algo = self.get_algorithm()

    def learn(self, name=None, tags=None, wandb_config={}, wandb_kwargs={}):
        """Learn the experiment."""
        if self.verbose > 0:
            pprint(self.config.dict())

        # Get the environment and algorithm
        env = self.env
        algo = self.algo

        # Start wandb
        run = wandb.init(
            project=self.project_name,
            config=self.config.dict() | wandb_config,
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
                    log_interval=self.config.log_interval,
                    tb_log_name=run_name)

        # Replace previous latest-model with the new model
        model.save(f"{self.MODELS_PATH}/latest-model")

        self.model = model

        self.evaluate(self.config.evaluate)

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
        elif self.config.algorithm.lower() == "td3":
            algo = TD3
        elif self.config.algorithm.lower() == "dsac":
            algo = DSAC
        else:
            raise ValueError(f"Algorithm {self.config.algorithm} not implemented.")
        return algo

    def load_model(self, model_name="olive-sun-4"):
        """Load a model file."""
        model = self.algo.load(Path.models / self.project_name / model_name / "model.zip")
        self.model = model

    def evaluate(self, n_times=1):
        """Run the experiment n times."""
        env = self.env

        for _ in range(n_times):
            obs = env.reset()

            for i in range(self.config.episode_steps):
                action, _states = self.model.predict(obs, deterministic=True)

                obs, reward, done, info = env.step(action)

                env.render()

                if wandb.run is not None:
                    wandb.log({"reward": reward,
                               "episode_step": i})
                    wandb.log({"reference": env.reference[-1],
                               "state": env.track[-1],
                               "episode_step": i})
                    wandb.log({"action": action,
                               "episode_step": i, })
                    wandb.log({"tracking_error": env.sq_error[-1],
                               "episode_step": i})

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

    def __del__(self):
        """Finish the wandb logging."""
        self.finish_wandb()


class Experiment:
    """An experiment that contains multiple sweeps"""

    def __init__(self, filename: str, file_path: str = None, offline=None):
        self.file_path = pl.Path(file_path) if file_path else Path.exp
        self.filename = filename + ".yaml" if not filename.endswith(".yaml") else filename
        self.sweeps = []

        # Extract data from dict config
        config_data = self.load_config()

        self.base_config = config_data.pop("config")
        self.project_name = config_data.pop("project_name")
        self.offline = offline if offline is not None else config_data.pop("offline")
        self.n_learning = config_data.pop("n_learning")
        self.sweeps_config = config_data
        self.build_sweeps()

    def load_config(self):
        """Load the config file."""
        file = self.file_path / self.filename
        if not file.exists():
            raise FileNotFoundError(f"File {file} not found.")
        with open(file) as f:
            return ConfigExperiment(**yaml.load(f, Loader=yaml.SafeLoader)).dict()

    def build_sweeps(self):
        self.sweeps = []

        for parameter in ["algorithm", "reward_type", "observation_type"]:
            if not self.sweeps_config[parameter]:
                self.sweeps_config[parameter] = [self.base_config[parameter]]

        keys = self.sweeps_config.keys()

        experiment_configurations = list(itertools.product(*self.sweeps_config.values()))

        # Create different seeds for each configuration to run
        for i in range(self.n_learning):
            # Create a sweep for each configuration
            for sweep in experiment_configurations:
                sweep_config = ConfigLinearAircraft(**dict(self.base_config, **dict(zip(keys, sweep))))
                self.sweeps.append(Sweep(config=sweep_config, project_name=self.project_name, offline=self.offline))

    def learn(self):
        print(f"Running {len(self.sweeps)} sweeps {self.n_learning} times each.")
        for sweep in self.sweeps:
            sweep.learn()
