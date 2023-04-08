"""Module that builds the hybrid IDHP-SAC agent."""
from copy import copy
from typing import Optional

from agents import BaseAgent
from agents.base_callback import ListCallback
from agents.callbacks import OnlineCallback, TensorboardCallback
from agents.idhp.idhp import IDHP
from agents.idhp_sac.policy import IDHPSACActor, IDHPSACPolicy
from agents.sac.sac import SAC
from helpers.paths import Path
from helpers.wandb_helpers import evaluate


class IDHPSAC(BaseAgent):
    """Class that implements the hybrid IDHP-SAC agent."""

    name = "IDHPSAC"

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
            device: Optional[str] = None,
            _init_setup_model: bool = True,
            sac_hidden_layers: list = None,
            idhp_hidden_layers: list = None,
    ):
        """Initialize the agent."""
        # Build the IDHP agent
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
            device=device,
            log_dir=log_dir,
            save_dir=save_dir,
        )

        self.sac = SAC(
            env_sac,
            learning_rate=learning_rate,
            verbose=verbose,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            batch_size=batch_size,
            policy_kwargs={"hidden_layers": sac_hidden_layers},
            device=device,
            log_dir=log_dir,
            save_dir=save_dir,
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
            _init_setup_model=_init_setup_model,
        )

    def setup_model(self):
        """Set up the model."""
        pass

    def _learn(
            self,
            total_steps: int,
            callback: ListCallback,
            log_interval: int,
            sac_steps: int = 1_000_000,
            idhp_steps: int = 1_000_000,
            sac_model: Optional[str] = None,
    ) -> None:
        """Learn the agent."""

        self.print("Offline learning")
        self.learn_offline(log_interval, sac_steps, sac_model)

        self.print("Online learning")
        self.learn_online(log_interval, idhp_steps)

        self.print("done 🎉")

    def learn_offline(self,
                      log_interval: int,
                      sac_steps: int,
                      sac_model: Optional[str], ):
        """Offline learning part of the algorithm."""
        if sac_model is not None:
            self.print("Loading SAC")
            sac_model_path = Path.models / sac_model
            self.sac.load(sac_model_path)
        else:
            self.print("Learning SAC")
            self.sac.learn(
                sac_steps,
                run_name="SAC",
                callback=[TensorboardCallback(verbose=self.verbose)],
                log_interval=log_interval,
            )

        # Evaluate SAC
        self.print("Evaluating SAC")
        evaluate(self.sac, self.sac.env)

    def learn_online(self,
                     log_interval: int,
                     idhp_steps: int = 1_000_000, ):
        """Online learning part of the algorithm."""
        self.print("Tranfering learning from SAC -> IDHP")

        self.idhp.policy.actor = self.policy.transfer_learning(self.sac, self.idhp)

        self.print("Learning IDHP")

        self.idhp.learn(
            idhp_steps,
            run_name="IDHP",
            callback=[
                OnlineCallback(verbose=self.verbose),
                TensorboardCallback(verbose=self.verbose),
            ],
            log_interval=log_interval,
        )

    def save(self, *args, **kwargs):
        """Save the agent."""
        # Give same run name for sac and idhp
        self.sac.run_name = self.run_name
        self.idhp.run_name = self.run_name
        self.sac.save(*args, **kwargs)
        self.idhp.save(*args, **kwargs)
        super().save(*args, **kwargs)
