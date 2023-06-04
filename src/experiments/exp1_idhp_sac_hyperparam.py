import os

import numpy as np
import wandb

from agents.config.config_idhp_dsac import (
    ConfigIDHPDSAC,
    ConfigIDHPDSACKwargs,
    ConfigIDHPDSACLearn,
)
from agents.config.config_idhp_sac import (
    ConfigIDHPKwargs,
    ConfigIDHPSAC,
    ConfigIDHPSACKwargs,
    ConfigIDHPSACLearn,
)
from agents.idhp_dsac.idhp_dsac import IDHPDSAC
from agents.idhp_sac.idhp_sac import IDHPSAC
from envs.citation.citation_env import CitationEnv
from envs.config.config_citation_env import ConfigCitationEnv, ConfigCitationKwargs
from helpers.paths import Path


os.environ["WANDB_DIR"] = str(Path().logs)
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

    agent_config = AgentConfig(
        kwargs=AgentKwargs(
            verbose=0,
            seed=np.random.randint(0, 1000),
            idhp_kwargs=ConfigIDHPKwargs(
                lr_a_high=config.lr_a_high,
                lr_c_high=config.lr_c_high,
                discount_factor=config.discount_factor,
                discount_factor_model=config.discount_factor_model,
            ),
        ),
        learn=AgentLearn(
            idhp_steps=4_000,
            sac_model=config.sac_model,
            callback=[],
        ),
    )

    agent = Agent(env=env, **agent_config.kwargs.dict())
    try:
        agent.learn(**agent_config.learn.dict())
    except ValueError:
        return 1e10, agent.sac_nmae * 100
    return agent.idhp_nmae * 100, agent.sac_nmae * 100


sweep_config = {
    "method": "random",
    "metric": {"name": "idhp_nmae", "goal": "minimize"},
    "parameters": {
        "lr_a_high": {
            "values": [
                0.00001,
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        },
        "lr_c_high": {
            "values": [
                0.00001,
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        },
        "task_train": {"values": ["exp1_pseudo_random_sin"]},
        "discount_factor": {"values": [0.6]},
        "discount_factor_model": {"values": [0.6]},
        "sac_model": {
            "values": [
                "SAC-citation/divine-grass-171",
                "SAC-citation/denim-leaf-172",
                "SAC-citation/firm-feather-173",
            ]
        },
        "agent": {"values": ["IDHPSAC"]},
    },
    "name": "changing-actor-critic",
}

sweep_config_dsac = {
    "method": "random",
    "metric": {"name": "idhp_nmae", "goal": "minimize"},
    "parameters": {
        "lr_a_high": {
            "values": [
                0.00001,
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        },
        "lr_c_high": {
            "values": [
                0.00001,
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        },
        "task_train": {"values": ["exp1_pseudo_random_sin"]},
        "discount_factor": {"values": [0.6]},
        "discount_factor_model": {"values": [0.6]},
        "sac_model": {
            "values": [
                "DSAC-citation/desert-fog-33",
                "DSAC-citation/smart-durian-34",
                "DSAC-citation/vague-hill-35",
            ]
        },
        "agent": {"values": ["IDHPDSAC"]},
    },
    "name": "changing-actor-critic-dsac",
}

sweep_config_actor = {
    "method": "random",
    "metric": {"name": "idhp_nmae", "goal": "minimize"},
    "parameters": {
        "lr_a_high": {
            "values": [
                0.00001,
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        },
        "lr_c_high": {"values": [0.001]},
        "task_train": {"values": ["exp1_pseudo_random_sin"]},
        "discount_factor": {"values": [0.6]},
        "discount_factor_model": {"values": [0.6]},
        "sac_model": {
            "values": [
                "SAC-citation/divine-grass-171",
                "SAC-citation/denim-leaf-172",
                "SAC-citation/firm-feather-173",
            ]
        },
        "agent": {"values": ["IDHPSAC"]},
    },
    "name": "fixed-critic-actor-changing",
}

sweep_config_actor_dsac = {
    "method": "random",
    "metric": {"name": "idhp_nmae", "goal": "minimize"},
    "parameters": {
        "lr_a_high": {
            "values": [
                0.00001,
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        },
        "lr_c_high": {"values": [0.001]},
        "task_train": {"values": ["exp1_pseudo_random_sin"]},
        "discount_factor": {"values": [0.6]},
        "discount_factor_model": {"values": [0.6]},
        "sac_model": {
            "values": [
                "DSAC-citation/desert-fog-33",
                "DSAC-citation/smart-durian-34",
                "DSAC-citation/vague-hill-35",
            ]
        },
        "agent": {"values": ["IDHPDSAC"]},
    },
    "name": "fixed-critic-actor-changing-dsac",
}

sweep_config_discount = {
    "method": "random",
    "metric": {"name": "idhp_nmae", "goal": "minimize"},
    "parameters": {
        "lr_a_high": {"values": [0.8]},
        "lr_c_high": {"values": [0.001]},
        "task_train": {"values": ["exp1_pseudo_random_sin"]},
        "discount_factor": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        },
        "discount_factor_model": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        },
        "sac_model": {
            "values": [
                "SAC-citation/divine-grass-171",
                "SAC-citation/denim-leaf-172",
                "SAC-citation/firm-feather-173",
            ]
        },
        "agent": {"values": ["IDHPSAC"]},
    },
    "name": "sac-learning-rates",
}

sweep_config_discount_dsac = {
    "method": "random",
    "metric": {"name": "idhp_nmae", "goal": "minimize"},
    "parameters": {
        "lr_a_high": {"values": [0.8]},
        "lr_c_high": {"values": [0.001]},
        "task_train": {"values": ["exp1_pseudo_random_sin"]},
        "discount_factor": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        },
        "discount_factor_model": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        },
        "sac_model": {
            "values": [
                "DSAC-citation/desert-fog-33",
                "DSAC-citation/smart-durian-34",
                "DSAC-citation/vague-hill-35",
            ]
        },
        "agent": {"values": ["IDHPDSAC"]},
    },
    "name": "sac-learning-rates-dsac",
}

def main():
    wandb.init(project="idhp-sac-hyperparams")
    idhp_nmae, sac_nmae = evaluate(wandb.config)
    wandb.log(
        {
            "idhp_nmae": idhp_nmae,
            "sac_nmae": sac_nmae,
            "nmae_improvement": sac_nmae - idhp_nmae,
        }
    )


sweep_id = wandb.sweep(sweep_config, project="idhp-sac-hyperparams")
wandb.agent(sweep_id, function=main, count=200)
