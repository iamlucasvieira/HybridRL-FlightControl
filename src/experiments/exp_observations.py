"""Experiment to analyse the difference in performance for different observation types."""

from experiments.core import Experiment
import numpy as np

experiment_kwargs = dict(
    project_name="citation-best-observation",
    seed=None,
    learning_steps=10_000,
    configuration="sp",
    task_name="q_sin",
    algorithm_name=None,
)


# 1: States + Reference + Error
# 2: Error
# 3: State + Reference[
# 4: States
# 3: State + Error
def get_value(item):
    return 0 if not item else item[-1]


for _ in range(10):

    for algo in ["SAC", "TD3", "DSAC"]:
        experiment_kwargs["algorithm_name"] = algo

        # Exp 1: Observation = all states + reference + tracking error
        exp_1 = Experiment(**experiment_kwargs)
        exp_1.learn(wandb_config={"obs": "states + ref + error"})
        exp_1.finish_wandb()

        # Exp 2: Observation = tracking error
        exp_2 = Experiment(**experiment_kwargs)
        env_2 = exp_2.env
        env_2._get_obs = lambda: np.array([get_value(env_2.sq_error)]).astype(np.float32)
        env_2.update_observation_space()
        exp_2.learn(wandb_config={"obs": "error"})
        exp_2.finish_wandb()

        # Exp 3: Observation = reference + state
        exp_3 = Experiment(**experiment_kwargs)
        env_3 = exp_3.env
        env_3._get_obs = lambda: np.array([get_value(env_3.reference), get_value(env_3.track)]).astype(np.float32)
        env_3.update_observation_space()
        exp_3.learn(wandb_config={"obs": "ref + state"})
        exp_3.finish_wandb()

        # Exp 4: Observation = all states
        exp_4 = Experiment(**experiment_kwargs)
        env_4 = exp_4.env
        env_4._get_obs = lambda: np.array(env_4.aircraft.current_state).astype(np.float32)
        env_4.update_observation_space()
        exp_4.learn(wandb_config={"obs": "states"})
        exp_4.finish_wandb()

        # Exp 5: Observation = state + error
        exp_5 = Experiment(**experiment_kwargs)
        env_5 = exp_5.env
        env_5._get_obs = lambda: np.array([get_value(env_5.track), get_value(env_5.sq_error)]).astype(np.float32)
        env_5.update_observation_space()
        exp_5.learn(wandb_config={"obs": "state + error"})
        exp_5.finish_wandb()

