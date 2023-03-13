"""Module that builds the hybrid IDHP-SAC agent."""
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback

from agents import IDHP, SAC
from agents.config import ConfigSAC, ConfigIDHP


class IDHPSAC(BaseAlgorithm):
    """Class that implements the hybrid IDHP-SAC agent."""

    policy_aliases = {"default": "MlpPolicy"}

    def __init__(self,
                 policy: str,
                 env: str,
                 learning_rate: float = 3e-4,
                 learning_starts: int = 100,
                 buffer_size: int = 1_000_000,
                 batch_size: int = 256,
                 policy_kwargs: dict = None,
                 tensorboard_log: str = None,
                 verbose: int = 1,
                 seed: int = 1,
                 device: str = "auto",
                 _init_setup_model: bool = True
                 ):
        """Initialize the agent."""
        self.idhp = IDHP(policy, env,
                         learning_rate=learning_rate,
                         verbose=verbose, )
        self.sac = SAC(policy, env,
                       learning_rate=learning_rate,
                       verbose=verbose,
                       learning_starts=learning_starts,
                       buffer_size=buffer_size,
                       batch_size=batch_size,)

        super(IDHPSAC, self).__init__(policy,
                                      env,
                                      learning_rate=learning_rate,
                                      policy_kwargs=policy_kwargs,
                                      tensorboard_log=tensorboard_log,
                                      verbose=verbose,
                                      seed=seed,
                                      device=device, )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        """Set up the model."""
        pass

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None,
              log_interval: int = 4,
              tb_log_name: str = "run",
              reset_num_timesteps: bool = True,
              progress_bar: bool = False,
              ) -> None:
        """Learn the agent."""
        self.print("Learning SAC")
        self.sac.learn(total_timesteps=total_timesteps,
                       callback=callback,
                       log_interval=log_interval,
                       tb_log_name=tb_log_name,
                       reset_num_timesteps=reset_num_timesteps,
                       progress_bar=progress_bar)
        self.print("done")

    def print(self, message):
        """Prints message based on verbosity."""
        if self.verbose > 0:
            print(message)
