import os

import numpy as np
import wandb

from agents.config.config_idhp_sac import (
    ConfigIDHPKwargs,
    ConfigIDHPSAC,
    ConfigIDHPSACKwargs,
    ConfigIDHPSACLearn,
)
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

    agent_config = ConfigIDHPSAC(
        kwargs=ConfigIDHPSACKwargs(
            verbose=0,
            seed=np.random.randint(0, 1000),
            idhp_kwargs=ConfigIDHPKwargs(
                lr_a_high=config.lr_a_high,
                lr_c_high=config.lr_c_high,
                discount_factor=config.discount_factor,
                discount_factor_model=config.discount_factor_model,
            ),
        ),
        learn=ConfigIDHPSACLearn(
            idhp_steps=4_000,
            sac_model="SAC-citation/divine-grass-171",
            callback=[],
        ),
    )

    agent = IDHPSAC(env=env, **agent_config.kwargs.dict())
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
        "discount_factor": {"values": [0.6, 0.7, 0.8, 0.9, 0.99]},
        "discount_factor_model": {"values": [0.6, 0.7, 0.8, 0.9, 0.99]},
    },
    "name": "learning_rates_v3",
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
    },
    "name": "fixed_critic_changing_actor",
}

sweep_config_actor_task = {
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
        "task_train": {
            "values": ["exp1_fixed_sin", "exp1_pseudo_random_sin", "exp1_hold"]
        },
        "discount_factor": {"values": [0.6]},
        "discount_factor_model": {"values": [0.6]},
    },
    "name": "fixed_critic_changing_actor_task",
}

sweep_config_discount = {
    "method": "random",
    "metric": {"name": "idhp_nmae", "goal": "minimize"},
    "parameters": {
        "lr_a_high": {"values": [0.3]},
        "lr_c_high": {"values": [0.001]},
        "task_train": {"values": ["exp1_pseudo_random_sin"]},
        "discount_factor": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        },
        "discount_factor_model": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        },
    },
    "name": "discount-factors",
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


sweep_id = wandb.sweep(sweep_config_discount, project="idhp-sac-hyperparams")
wandb.agent(sweep_id, function=main, count=200)
