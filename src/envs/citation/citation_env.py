"""Creates a gym environment for the high fidelity citation  model."""
from envs.base_env import BaseEnv


class CitationEnv(BaseEnv):
    """Citation Environment for the nonlinear citation model that follows gym interface"""

    def __init__(self, mode="default", configuration="sp", dt=0.1, episode_steps=100):
        super(CitationEnv, self).__init__()

    def reset(self):
        pass

    def step(self, action):
        pass

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
