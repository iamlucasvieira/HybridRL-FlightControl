"""Module that allows building an experiment based on yaml configutaion."""
import itertools
import pathlib as pl
import random
import torch
import operator

import wandb
import yaml
from rich import print

from helpers.misc import get_name
from helpers.paths import Path
from helpers.config_auto import validate_auto
from hrl_fc.experiment_config import ConfigExperiment
from helpers.callbacks import AVAILABLE_CALLBACKS


class Sweep:
    """Class that builds an experiment."""

    def __init__(self, config: ConfigExperiment):
        """Build a sweep."""
        self.config = config
        self.run_name = None

        # Define project name and paths
        if self.config.name is None:
            agent_name = self.agent_config.name
            env_name = self.env_config.name
            self.config.name = f"{agent_name}-{env_name}"

        self.MODELS_PATH = Path.models / self.config.name
        self.LOGS_PATH = Path.logs / self.config.name

        # Build env
        self.env = self.build_env()

        # Build agent
        self.agent = self.build_agent()

        # Learn kwargs
        self.learn_kwargs = self.agent_config.learn.dict()

        # Get callbacks
        if "callback" in self.learn_kwargs:
            self.get_callbacks()

    @property
    def agent_config(self):
        """Alias for agent config."""
        return self.config.agent.__root__

    @property
    def env_config(self):
        """Alias for env config."""
        return self.config.env

    def build_agent(self):
        agent_args = tuple(self.replace_auto_config(self.agent_config.args.dict()).values())
        agent_kwargs = self.replace_auto_config(self.agent_config.kwargs.dict())

        agent = self.agent_config.object(*agent_args, **agent_kwargs)

        return agent

    def build_env(self):
        env_kwargs = self.env_config.kwargs.dict()
        env = self.env_config.object(**env_kwargs)
        return env

    def replace_auto_config(self, config: dict) -> dict:
        """Replace configs set to auto by the required value"""

        def replace(key, new_value):
            if key in config and validate_auto(config[key]):
                config[key] = new_value

        replace("env", self.env)
        replace("verbose", self.config.verbose)
        replace("tensorboard_log", self.LOGS_PATH)
        replace("seed", self.config.seed)
        return config

    def get_callbacks(self):
        """Get the callbacks to use."""
        callbacks = []
        config_callbacks = self.learn_kwargs['callback']
        if "wandb" in config_callbacks:
            callbacks.append(AVAILABLE_CALLBACKS['wandb'](
                model_save_freq=100,
                model_save_path=f"{self.MODELS_PATH / self.run_name}",
                verbose=self.config.verbose
            ))

        if "tensorboard" in config_callbacks:
            callbacks.append(AVAILABLE_CALLBACKS['tensorboard'](
                verbose=self.config.verbose,
            ))

        if "online" in config_callbacks:
            callbacks.append(AVAILABLE_CALLBACKS['online'](
                verbose=self.config.verbose,
            ))

        self.learn_kwargs["callback"] = callbacks

    def learn(self, name=None):
        """Learn the experiment."""
        if name is None:
            name = get_name([self.config.name, self.agent_config.name])
        self.run_name = name

        # Create directories to save models and logs
        pl.Path.mkdir(self.MODELS_PATH / name, parents=True, exist_ok=True)
        pl.Path.mkdir(self.LOGS_PATH, parents=True, exist_ok=True)

        # Learn
        self.agent.learn(**self.learn_kwargs)

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


class ExperimentBuilder:
    """Class that builds an experiment from a config."""

    def __init__(self, filename: str, file_path: str = None):

        self.file_path = pl.Path(file_path) if file_path else Path.exp
        self.filename = filename + ".yaml" if not filename.endswith(".yaml") else filename
        self.sweeps = []
        self.sweep_configs = []

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
        # Give default values for items that have no sweep set.
        def get_sweep(sweep_attr: str, default_attr: str):
            """Gets all possible sweeps."""
            # If values from sweep are empty, initialize them with the default values.
            sweep_list = []
            sweep = operator.attrgetter(sweep_attr)(self.config)
            defaults = operator.attrgetter(default_attr)(self.config)

            for sweep_option, sweep_value in sweep.dict().items():
                if not isinstance(sweep_value, list):
                    setattr(sweep, sweep_option, [getattr(defaults, sweep_option)])

            sweep_configurations = list(itertools.product(*sweep.dict().values()))

            for sweep_config in sweep_configurations:
                config_list = [self.config] if len(self.sweep_configs) == 0 else self.sweep_configs

                # Loop through existing sweeps
                for config in config_list:
                    new_config = config.copy(deep=True)
                    sweep_config_dict = dict(zip(sweep.dict().keys(), sweep_config))
                    operator.attrgetter(default_attr)(new_config).__dict__.update(sweep_config_dict)
                    sweep_list.append(new_config)
            return sweep_list

        # Get all possible sweeps
        self.sweep_configs = get_sweep("env.sweep", "env.kwargs")
        self.sweep_configs = get_sweep("agent.__root__.sweep", "agent.__root__.kwargs")

        for _ in range(self.config.n_learning):
            for sweep_config in self.sweep_configs:
                self.sweeps.append(Sweep(sweep_config))

        # If multiple sweeps generate seeds
        if len(self.sweeps) > 1:
            if self.config.seed is not None:
                random.seed(self.config.seed)
            for sweep in self.sweeps:
                sweep.config.seed = self.get_random_seed()

    def get_random_seed(self):
        """Get a random seed."""
        return random.randint(0, 2 ** 32 - 1)