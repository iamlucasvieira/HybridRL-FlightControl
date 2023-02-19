"""File that defines the core of the experiments."""
import itertools
import os
import pathlib as pl
import random
import torch

import matplotlib.pyplot as plt
import wandb
import yaml
from rich import print
from rich.progress import track
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from helpers.misc import get_name
from helpers.paths import Path, set_wandb_path
from helpers.tracking import TensorboardCallback
from hrl_fc.config import ConfigExperiment


class Sweep:
    """Class that builds an experiment."""

    def __init__(self, config: ConfigExperiment = None, **kwargs):
        """Initiates the experiment.

        args:
            config: configuration of the experiment
            algorithm_name: Name of the algorithm used.
            env_name: Environment to use.
            filename: Name of the file to load the aircraft configuration from.
            task_type: Task to perform.
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
            MODELS_PATH: Path to the envs.
            LOGS_PATH: Path to the logs.
            wandb_run: Wandb run.
        """

        if config is None:
            self.config = ConfigExperiment(**kwargs)
        else:
            self.config = config

        if self.config.offline:
            os.environ["WANDB_MODE"] = "offline"

        # Set the wandb log path
        set_wandb_path()

        self.model = None
        self.wandb_run = None

        # Define project name and paths
        if self.config.name is None:
            agent_name = self.config.agent.__root__.name
            env_name = self.config.env.name
            task_type = self.config.env.config.task_type
            self.config.name = f"{agent_name}-{env_name}-{task_type}"

        self.MODELS_PATH = Path.models / self.config.name
        self.LOGS_PATH = Path.logs / self.config.name

        self.env = self.config.env.object(config=self.config.env.config)
        self.algo = self.config.agent.__root__.object

    def learn(self, name=None, tags=None, wandb_config={}, wandb_kwargs={}):
        """Learn the experiment."""
        if self.config.verbose > 0:
            print(f"Starting experiment {self.config.name}")
            print(self.config.dict())

        # Get the environment and algorithm
        env = self.env
        algo = self.algo

        # Start wandb
        run = wandb.init(
            project=self.config.name,
            config=self.config.dict() | wandb_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,  # optional
            name=name,
            tags=tags,
            **wandb_kwargs,
        )

        self.wandb_run = run

        # If online, get the run name provided by wandb
        run_name = get_name([self.config.name, self.config.agent.__root__.name]) if self.config.offline else run.name

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
        algo_kwargs = self.config.agent.__root__.config.dict()
        model = algo(
            algo_kwargs.pop("policy"),
            env,
            verbose=self.config.verbose,
            tensorboard_log=self.LOGS_PATH,
            seed=self.config.seed,
            **algo_kwargs, )

        # Learn model
        model.learn(total_timesteps=self.config.learning_steps,
                    callback=[wandb_callback, tensorboard_callback],
                    log_interval=self.config.log_interval,
                    tb_log_name=run_name)

        # Replace previous latest-model with the new model
        # model.save(f"{self.MODELS_PATH}/latest-model")

        self.model = model

        self.evaluate(self.config.evaluate)

    def load_model(self, model_name="olive-sun-4"):
        """Load a model file."""
        model = self.algo.load(Path.models / self.project_name / model_name / "model.zip")
        self.model = model

    def evaluate(self, n_times=1):
        """Run the experiment n times."""
        env = self.env

        for _ in range(n_times):
            obs = env.reset()

            for i in range(self.config.env.config.episode_steps):
                action, _states = self.model.predict(obs, deterministic=True)

                # Transform action into numpy if it is a tensor
                if isinstance(action, torch.Tensor):
                    action = action.detach().numpy()

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
    """Class that builds an experiment from a config."""

    def __init__(self, filename: str, file_path: str = None, offline=None):

        self.file_path = pl.Path(file_path) if file_path else Path.exp
        self.filename = filename + ".yaml" if not filename.endswith(".yaml") else filename
        self.sweeps = []

        # Extract data from dict config
        self.config = self.load_config()

        self.build_sweeps()

    def load_config(self):
        """Load the config file."""
        file = self.file_path / self.filename
        if not file.exists():
            raise FileNotFoundError(f"File {file} not found.")
        with open(file) as f:
            return ConfigExperiment(**yaml.load(f, Loader=yaml.SafeLoader))

    def build_sweeps(self):
        self.sweeps = []

        # Get list of agents
        agent_list = self.config.agent if isinstance(self.config.agent, list) else [self.config.agent]

        # Give default values for items that have no sweep set.
        for sweep_option, sweep_value in self.config.env.sweep.dict().items():
            if len(sweep_value) == 0:
                getattr(self.config.env.sweep, sweep_option).append(getattr(self.config.env.config, sweep_option))

        for agent in agent_list:

            sweep_configurations = list(itertools.product(*self.config.env.sweep.dict().values()))

            for _ in range(self.config.n_learning):
                for sweep_config in sweep_configurations:
                    config = self.config.copy(deep=True)
                    config.agent = agent
                    sweep_config_dict = dict(zip(config.env.sweep.dict().keys(), sweep_config))
                    config.env.config.__dict__.update(sweep_config_dict)

                    self.sweeps.append(Sweep(config=config))

            # If multiple sweeps generate seeds
            if self.config.seed is not None:
                random.seed(self.config.seed)

            for sweep in self.sweeps:
                sweep.config.seed = self.get_random_seed()

    def learn(self):
        """Run the experiment."""
        if self.config.verbose > 0:
            print(f"Running {len(self.sweeps)} sweeps")
        for sweep in track(self.sweeps, description=":brain: Learning..."):
            sweep.learn()

    def get_random_seed(self):
        """Get a random seed."""
        return random.randint(0, 2 ** 32 - 1)
