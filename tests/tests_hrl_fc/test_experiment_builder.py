import pathlib as pl

import pytest

from helpers.paths import Path
from hrl_fc.experiment_builder import ExperimentBuilder, Sweep
from hrl_fc.experiment_config import ConfigExperiment


TESTS_PATH = pl.Path(__file__).parent


class TestSweep:
    """Test the Sweep class."""

    def test_sweep_name(self):
        """Test that the sweep name is correct."""
        sweep_name = "test"
        config_no_name = ConfigExperiment()
        config_w_name = ConfigExperiment(name=sweep_name)

        sweep_no_name = Sweep(config_no_name)
        sweep_w_name = Sweep(config_w_name)

        assert sweep_no_name.config.name != sweep_name
        assert sweep_w_name.config.name == sweep_name

    def test_sweep_path(self):
        """Test that the sweep paths are correct."""
        sweep_name = "test"
        config = ConfigExperiment(name=sweep_name)
        sweep = Sweep(config)

        assert sweep.MODELS_PATH == Path.models / sweep_name
        assert sweep.LOGS_PATH == Path.logs / sweep_name


class TestExperimentBuilder:
    def test_experiment_builder_no_file(self):
        assert 1 == 1
