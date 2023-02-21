"""Class the creates the idhp algorithm."""
from IDHP.networks import CriticNetwork, ActorNetwork
import gym


class IDHP:
    """IDHP class."""

    def __init__(self, env: gym.env):
        """Initialize IDHP."""
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.critic = CriticNetwork()
        self.target_critic = CriticNetwork()

        self.actor = ActorNetwork()

