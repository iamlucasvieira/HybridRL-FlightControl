"""Module that tests the DSAC agent."""
from typing import Type

import pytest

from agents import DSAC
from agents.dsac.policy import DSACPolicy
from envs import BaseEnv, CitationEnv, LTIEnv


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestDSAC:
    """Class that tests the DSAC objct."""

    def test_init(self, env: Type[BaseEnv]):
        """Tests the initialization of the DSAC object."""
        agent = DSAC(env())
        assert isinstance(agent, DSAC)
        assert isinstance(agent.policy, DSACPolicy)
        assert isinstance(agent.target_policy, DSACPolicy)

    def test_update(self, env: Type[BaseEnv]):
        """Tests the update method of the DSAC object."""
        pass
        # agent = DSAC(env())
        # agent.update()
        # assert agent._n_updates == 1
