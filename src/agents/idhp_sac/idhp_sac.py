"""Module that builds the hybrid IDHP-SAC agent."""
from copy import copy
from typing import List, Tuple, Optional

from agents.base_agent import BaseAgent
from agents import IDHP, SAC
from agents.idhp_sac.policy import IDHPSACPolicy, IDHPSACActor
from helpers.sb3 import load_agent
from helpers.torch_helpers import get_device
from helpers.wandb_helpers import evaluate


class IDHPSAC(BaseAgent):
    """Class that implements the hybrid IDHP-SAC agent."""

    policy_aliases = {"default": IDHPSACPolicy}

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
                 device: str = None,
                 _init_setup_model: bool = True,
                 sac_hidden_layers: list = None,
                 idhp_hidden_layers: list = None,
                 ):
        """Initialize the agent."""
        # Build the IDHP agent
        if device is None:
            device = get_device()

        if sac_hidden_layers is None:
            sac_hidden_layers = [256, 256]

        if idhp_hidden_layers is None:
            idhp_hidden_layers = [10, 10]

        actor_kwargs = {"hidden_layers": idhp_hidden_layers}
        critic_kwargs = {"hidden_layers": idhp_hidden_layers}
        # Make sure environment follows IDHP requirements
        env = IDHP._setup_env(env)

        # Make copies of env for SAC and IDHP
        env_sac, env_idhp = copy(env), copy(env)
        self.idhp = IDHP("default", env_idhp,
                         learning_rate=learning_rate,
                         verbose=verbose,
                         actor_kwargs=actor_kwargs,
                         critic_kwargs=critic_kwargs, )

        self.sac = SAC("default", env_sac,
                       learning_rate=learning_rate,
                       verbose=verbose,
                       learning_starts=learning_starts,
                       buffer_size=buffer_size,
                       batch_size=batch_size,
                       policy_kwargs={"hidden_layers": sac_hidden_layers}, )

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

    def _setup_idhp(self):
        """Set up the IDHP model."""
        # The first hidden layers of the idhp should be the layers of the SAC
        idhp_actor = self.idhp.policy.actor
        sac_actor = self.sac.policy.actor

        # Update the IDHP layers
        self.idhp.policy.actor = IDHPSACActor(idhp_actor, sac_actor)

    def _setup_model(self):
        """Set up the model."""
        pass

    def learn(self,
              sac_timesteps: int = 1_000_000,
              idhp_timesteps: int = 1_000_000,
              sac_model: Optional[str] = None,
              callback = None,
              log_interval: int = 4,
              tb_log_name: str = "run",
              reset_num_timesteps: bool = True,
              progress_bar: bool = False,
              ) -> None:
        """Learn the agent."""

        if sac_model is not None:
            self.print("Loading SAC")
            self.sac = load_agent(sac_model).sac
        else:
            self.print("Learning SAC")
            self.sac.learn(total_timesteps=sac_timesteps,
                           callback=[TensorboardCallback(verbose=self.verbose)],
                           log_interval=log_interval,
                           tb_log_name=tb_log_name,
                           reset_num_timesteps=reset_num_timesteps,
                           progress_bar=progress_bar)

        # Evaluae SAC
        self.print("Evaluating SAC")
        evaluate(self.sac, self.sac._env)

        self.print("Tranfering learning from SAC -> IDHP")
        self._setup_idhp()

        self.print("Learning IDHP")
        self.idhp.learn(total_timesteps=idhp_timesteps,
                        callback=[OnlineCallback(verbose=self.verbose), TensorboardCallback(verbose=self.verbose)],
                        log_interval=log_interval,
                        tb_log_name=tb_log_name,
                        reset_num_timesteps=reset_num_timesteps,
                        progress_bar=progress_bar)
        self.print("done 🎉")

    @property
    def _env(self):
        """Return the environment."""
        return self.env

    def print(self, message):
        """Prints message based on verbosity."""
        if self.verbose > 0:
            print(message)

    def _excluded_save_params(self) -> List[str]:
        default_excluded_params = super()._excluded_save_params()
        if 'env' in default_excluded_params:
            default_excluded_params.remove('env')

        return default_excluded_params + ["idhp", "sac"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """Return the parameters to save."""
        return ["sac.policy"], []
