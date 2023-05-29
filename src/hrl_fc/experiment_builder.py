"""Module that allows building an experiment based on yaml configutaion."""
import itertools
import operator
import pathlib as pl
import random
import shutil
from typing import Optional

import yaml

from agents import AVAILABLE_CALLBACKS, BaseAgent
from helpers.config_auto import validate_auto
from helpers.misc import get_name
from helpers.paths import Path
from helpers.wandb_helpers import evaluate
from hrl_fc.experiment_config import ConfigExperiment


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

    @property
    def agent_config(self):
        """Alias for agent config."""
        return self.config.agent

    @property
    def env_config(self):
        """Alias for env config."""
        return self.config.env

    def build_agent(self) -> BaseAgent:
        agent_args = tuple(
            self.replace_auto_config(self.agent_config.args.dict()).values()
        )
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

        # Agent args and kwargs
        replace("env", self.env)
        replace("verbose", self.config.verbose)
        replace("log_dir", self.LOGS_PATH)
        replace("save_dir", self.MODELS_PATH)
        replace("seed", self.config.seed)

        # Learn kwargs
        replace("run_name", self.run_name)
        return config

    def get_callbacks(self):
        """Get the callbacks to use."""
        callbacks = []

        for callback in self.learn_kwargs["callback"]:
            if isinstance(callback, tuple):
                name, kwargs = callback
            else:
                name, kwargs = callback, {}

            if name not in AVAILABLE_CALLBACKS:
                raise ValueError(f"Callback {callback} not available.")
            else:
                callbacks.append(
                    AVAILABLE_CALLBACKS[name](verbose=self.config.verbose, **kwargs)
                )

        self.learn_kwargs["callback"] = callbacks

    def learn(self, name=None, progress=None):
        """Learn the experiment."""
        if name is None:
            name = get_name([self.config.name, self.agent_config.name])
        self.run_name = name

        # Get callbacks
        if "callback" in self.learn_kwargs:
            self.get_callbacks()

        # Create directories to save models and logs
        pl.Path.mkdir(self.MODELS_PATH / name, parents=True, exist_ok=True)
        pl.Path.mkdir(self.LOGS_PATH, parents=True, exist_ok=True)

        # Replace auto config
        learn_kwargs = self.replace_auto_config(self.learn_kwargs)

        # Learn
        self.agent.learn(**learn_kwargs)

    def load_model(self, model_path: Optional[pl.Path], run: str = "best"):
        """Load a model file."""
        self.agent.load(model_path, run=run)

    def save_model(self, config_path: Optional[pl.Path] = None):
        """Save a model file."""
        self.agent.save()
        # Copy config file to model folder
        if config_path is not None:
            shutil.copy(config_path, self.MODELS_PATH / self.run_name / "config.yaml")

    def evaluate(self, config_path: Optional[pl.Path] = None):
        """Evaluate the agent."""
        # Load best model
        if config_path is not None:
            self.agent.load(path=self.agent.save_dir / self.agent.run_name, run="best")
        evaluate(self.agent, self.env, n_times=self.config.evaluate)


class ExperimentBuilder:
    """Class that builds an experiment from a config."""

    def __init__(self, filename: str, file_path: str = None):
        self.file_path = pl.Path(file_path) if file_path else Path.exp
        self.filename = (
            filename + ".yaml" if not filename.endswith(".yaml") else filename
        )
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
                if not isinstance(sweep_value, list) or sweep_option == "hidden_layers":
                    setattr(sweep, sweep_option, [getattr(defaults, sweep_option)])

            sweep_configurations = list(itertools.product(*sweep.dict().values()))

            for sweep_config in sweep_configurations:
                config_list = (
                    [self.config]
                    if len(self.sweep_configs) == 0
                    else self.sweep_configs
                )

                # Loop through existing sweeps
                for config in config_list:
                    new_config = config.copy(deep=True)
                    sweep_config_dict = dict(zip(sweep.dict().keys(), sweep_config))
                    operator.attrgetter(default_attr)(new_config).__dict__.update(
                        sweep_config_dict
                    )
                    sweep_list.append(new_config)
            return sweep_list

        # Get all possible sweeps
        self.sweep_configs = get_sweep("env.sweep", "env.kwargs")
        self.sweep_configs = get_sweep("agent.sweep", "agent.kwargs")

        MULTIPLE_SWEEPS = len(self.sweep_configs) > 1 or self.config.n_learning > 1
        if MULTIPLE_SWEEPS and self.config.seed is not None:
            random.seed(self.config.seed)
        for _ in range(self.config.n_learning):
            for sweep_config in self.sweep_configs:
                if MULTIPLE_SWEEPS:
                    sweep_config.seed = self.get_random_seed()
                elif self.config.seed is None:
                    sweep_config.seed = self.get_random_seed()
                self.sweeps.append(Sweep(sweep_config))

    def get_random_seed(self):
        """Get a random seed."""
        return random.randint(0, 1_000_000)
