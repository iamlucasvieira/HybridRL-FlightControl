"""Module that tests the DSAC agent."""
import pytest

from agents import DSAC
from envs import BaseEnv
from envs import CitationEnv
from envs import LTIEnv


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestDSAC:
    """Class that tests the DSAC objct."""

    def test_init(self, env: BaseEnv):
        """Tests the initialization of the DSAC object."""
        agent = DSAC(env())
        assert agent is not None
