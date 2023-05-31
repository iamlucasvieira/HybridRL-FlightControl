from agents.config.config_idhp_sac import ConfigIDHPSAC, ConfigIDHPSACKwargs, ConfigIDHPSACLearn, ConfigIDHPKwargs
from agents.config.config_idhp_dsac import ConfigIDHPDSAC, ConfigIDHPDSACKwargs, ConfigIDHPDSACLearn
from envs.config.config_citation_env import ConfigCitationEnv, ConfigCitationKwargs

from agents.idhp_sac.idhp_sac import IDHPSAC
from agents.idhp_dsac.idhp_dsac import IDHPDSAC
from envs.citation.citation_env import CitationEnv
import numpy as np
import wandb

np.random.seed(2)


def evaluate(config):
    task_train = "att_eval" if not hasattr(config, "task_train") else config.task_train

    env_config = ConfigCitationEnv(
        kwargs=ConfigCitationKwargs(
            dt=0.01,
            episode_steps=4_000,
            eval_steps=4_000,
            task_train=task_train,
            reward_type="clip",
            observation_type="sac_attitude",
        )
    )
    env = CitationEnv(**env_config.kwargs.dict())

    if config.agent == "IDHPSAC":
        AgentConfig = ConfigIDHPSAC
        AgentKwargs = ConfigIDHPSACKwargs
        AgentLearn = ConfigIDHPSACLearn
        Agent = IDHPSAC
    elif config.agent == "IDHPDSAC":
        AgentConfig = ConfigIDHPDSAC
        AgentKwargs = ConfigIDHPDSACKwargs
        AgentLearn = ConfigIDHPDSACLearn
        Agent = IDHPDSAC
    else:
        raise ValueError(f"Unknown agent {config.agent}")

    hybrid_config = AgentConfig(
        kwargs=AgentKwargs(
            verbose=0,
            seed=np.random.randint(0, 1000),
            idhp_kwargs=ConfigIDHPKwargs(
                lr_a_high=config.lr_a_high,
                lr_c_high=config.lr_c_high,
            )
        ),
        learn=AgentLearn(
            idhp_steps=4_000,
            sac_model=config.sac_model,
            callback=[],
        ),

    )
    hybrid_agent = Agent(env=env, **hybrid_config.kwargs.dict())

    results = {}
    try:
        hybrid_agent.learn(**hybrid_config.learn.dict())
        results["sac_hybrid"] = hybrid_agent.idhp_nmae * 100
        results["sac_only"] = hybrid_agent.sac_nmae * 100
    except ValueError:
        results["sac_hybrid"] = 1e10
        results["sac_only"] = hybrid_agent.sac_nmae * 100

    results["sac_improvement"] = results["sac_only"] - results["sac_hybrid"]

    return results


sweep_config_sac = {
    "method": "grid",
    "parameters": {
        "lr_a_high": {"values": [0.8]},
        "lr_c_high": {"values": [0.3]},
        "task_train": {"values": ["exp1_hold", "exp1_fixed_sin", "exp1_pseudo_random_sin"]},
        "sac_model": {"values": ["SAC-citation/divine-grass-171", "SAC-citation/denim-leaf-172",
                                 "SAC-citation/firm-feather-173"]},
        "seed": {"values": [1, 2, 3, 4, 5]},
        "algorithm": {"values": ["IDHPSAC"]},
    }
}

sweep_config_dsac = sweep_config_sac.copy()
sweep_config_dsac["parameters"]["sac_model"] = {
    "values": ["DSAC-citation/icy-meadow-174", "DSAC-citation/icy-meadow-175", "DSAC-citation/icy-meadow-176"]}
sweep_config_dsac["parameters"]["algorithm"] = {"values": ["IDHPDSAC"]}


def main():
    wandb.init(project="exp1_test")
    results = evaluate(wandb.config)
    wandb.log(results)


sweep_id = wandb.sweep(sweep_config, project="exp1_test")
wandb.agent(sweep_id, function=main)
