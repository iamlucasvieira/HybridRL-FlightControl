"""Module that builds the hybrid IDHP-SAC agent."""
from copy import copy
from typing import Optional

from agents import IDHP
from agents import SAC
from agents import BaseAgent
from agents.base_callback import ListCallback
from agents.callbacks import OnlineCallback
from agents.callbacks import TensorboardCallback
from agents.idhp_sac.policy import IDHPSACActor
from agents.idhp_sac.policy import IDHPSACPolicy
from helpers.sb3 import load_agent
from helpers.torch_helpers import get_device
from helpers.wandb_helpers import evaluate


class IDHPSAC(BaseAgent):
    """Class that implements the hybrid IDHP-SAC agent."""

    def __init__(
        self,
        env: str,
        learning_rate: float = 3e-4,
        learning_starts: int = 100,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        policy_kwargs: dict = None,
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
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
        self.idhp = IDHP(
            env_idhp,
            learning_rate=learning_rate,
            verbose=verbose,
            actor_kwargs=actor_kwargs,
            critic_kwargs=critic_kwargs,
        )

        self.sac = SAC(
            env_sac,
            learning_rate=learning_rate,
            verbose=verbose,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            batch_size=batch_size,
            policy_kwargs={"hidden_layers": sac_hidden_layers},
        )

        super().__init__(
            IDHPSACPolicy,
            env,
            log_dir=log_dir,
            save_dir=save_dir,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        if _init_setup_model:
            self._setup_model()

    def setup_model(self):
        """Set up the model."""
        pass

    def _setup_idhp(self):
        """Set up the IDHP model."""
        # The first hidden layers of the idhp should be the layers of the SAC
        idhp_actor = self.idhp.policy.actor
        sac_actor = self.sac.policy.actor

        # Update the IDHP layers
        self.idhp.policy.actor = IDHPSACActor(idhp_actor, sac_actor)

    def _learn(
        self,
        total_steps: int,
        callback: ListCallback,
        log_interval: int,
        sac_timesteps: int = 1_000_000,
        idhp_timesteps: int = 1_000_000,
        sac_model: Optional[str] = None,
    ) -> None:
        """Learn the agent."""

        if sac_model is not None:
            self.print("Loading SAC")
            self.sac = load_agent(sac_model).sac
        else:
            self.print("Learning SAC")
            self.sac.learn(
                sac_timesteps,
                run_name="SAC",
                callback=[TensorboardCallback(verbose=self.verbose)],
                log_interval=log_interval,
            )

        # Evaluae SAC
        self.print("Evaluating SAC")
        evaluate(self.sac, self.sac.env)

        self.print("Tranfering learning from SAC -> IDHP")
        self._setup_idhp()

        self.print("Learning IDHP")
        self.idhp.learn(
            idhp_timesteps,
            run_name="IDHP",
            callback=[
                OnlineCallback(verbose=self.verbose),
                TensorboardCallback(verbose=self.verbose),
            ],
            log_interval=log_interval,
        )
        self.print("done 🎉")
