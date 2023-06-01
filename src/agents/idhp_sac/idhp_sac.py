"""Module that builds the hybrid IDHP-SAC agent."""
from copy import copy
from typing import Optional

from agents import BaseAgent
from agents.base_callback import ListCallback
from agents.callbacks import IDHPSACCallback, OnlineCallback, TensorboardCallback
from agents.idhp.idhp import IDHP
from agents.idhp_sac.policy import IDHPSACPolicy
from agents.sac.sac import SAC
from helpers.paths import Path
from helpers.wandb_helpers import evaluate


class IDHPSAC(BaseAgent):
    """Class that implements the hybrid IDHP-SAC agent."""

    name = "IDHPSAC"

    def __init__(
        self,
        env: str,
        policy_kwargs: dict = None,
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        verbose: int = 1,
        seed: int = 1,
        device: Optional[str] = None,
        _init_setup_model: bool = True,
        idhp_kwargs: dict = None,
        sac_kwargs: dict = None,
        idhp_actor_observation: str = "sac_attitude",
    ):
        """Initialize the agent."""
        # Build the IDHP agent
        idhp_kwargs = {} if idhp_kwargs is None else idhp_kwargs
        sac_kwargs = {} if sac_kwargs is None else sac_kwargs

        for agent_dict in [idhp_kwargs, sac_kwargs]:
            agent_dict["log_dir"] = log_dir
            agent_dict["save_dir"] = save_dir
            agent_dict["verbose"] = verbose
            agent_dict["seed"] = seed

        # Make sure environment follows IDHP requirements
        idhp_kwargs["actor_observation_type"] = idhp_actor_observation

        # Make copies of env for SAC and IDHP
        env_sac = copy(env)
        env_idhp = IDHP._setup_env(env)

        self.idhp = IDHP(
            env_idhp,
            **idhp_kwargs,
        )

        self.sac = SAC(
            env_sac,
            **sac_kwargs,
        )

        self.sac_nmae = None
        self.idhp_nmae = None

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

    def learn_offline(
        self,
        log_interval: int,
        sac_steps: int,
        sac_model: Optional[str],
    ):
        """Offline learning part of the algorithm."""
        if sac_model is not None:
            self.print("Loading SAC")
            sac_model_path = Path.models / sac_model
            self.sac.load(sac_model_path, run="best")
        else:
            self.print("Learning SAC")
            self.sac.learn(
                sac_steps,
                run_name="SAC",
                callback=[TensorboardCallback(verbose=self.verbose)],
                log_interval=log_interval,
            )

    def learn_online(
        self,
        log_interval: int,
        idhp_steps: int = 1_000_000,
    ):
        """Online learning part of the algorithm."""

        # self.sac.env.set_observation_function("noise + states + ref")
        # self.idhp.env.set_observation_function("noise + states + ref")

        # Evaluate SAC
        self.print("Evaluating SAC")
        _, sac_nmae = evaluate(self.sac, self.sac.env, to_wandb=True)
        self.sac_nmae = sac_nmae

        self.print("Tranfering learning from SAC -> IDHP")
        self.idhp.policy.actor = self.policy.transfer_learning(self.sac, self.idhp)

        self.print("Learning IDHP")

        self.idhp.learn(
            idhp_steps,
            run_name="IDHP",
            callback=[
                OnlineCallback(verbose=self.verbose),
                TensorboardCallback(verbose=self.verbose),
                IDHPSACCallback(verbose=self.verbose),
            ],
            log_interval=log_interval,
        )

        self.logger.record("nMAE_sac", sac_nmae * 100)
        self.logger.record("nMAE_idhp", self.idhp.env.nmae * 100)
        self.logger.dump()

        self.idhp_nmae = self.idhp.env.nmae

    def save(self, *args, **kwargs):
        """Save the agent."""
        # Give same run name for sac and idhp
        self.sac.run_name = self.run_name
        self.idhp.run_name = self.run_name
        self.sac.save(*args, **kwargs)
        self.idhp.save(*args, **kwargs)
        super().save(*args, **kwargs)

    def load(self, *args, **kwargs):
        pass
