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
    run=1
)


# Exp 1: Observation = all states + reference + tracking error
# Exp 2: Observation = tracking error
# Exp 3: Observation = reference + state
# Exp 4: Observation = all states
# Exp 5: Observation = state + error
def get_value(item):
    return 0 if not item else item[-1]

observation_dict = {
    "states + ref + error": None,
    # "error": lambda env: np.array([get_value(env.sq_error)]).astype(np.float32),
    "ref + state": lambda env: np.array([get_value(env.reference), get_value(env.track)]).astype(np.float32),
    "states": lambda env: np.array(env.aircraft.current_state).astype(np.float32),
    "state + error": lambda env: np.array([get_value(env.track), get_value(env.sq_error)]).astype(np.float32)
}

np.random.seed(1)


for _ in range(10):

    for algo in ["SAC", "TD3", "DSAC"]:

        experiment_kwargs["algorithm_name"] = algo
        experiment_kwargs["seed"] = np.random.randint(0,9e3)


        for obs, obs_function in observation_dict.items():

            exp = Experiment(**experiment_kwargs)

            if obs_function is not None:
                env = exp.env
                env._get_obs = lambda: obs_function(env)
                env.update_observation_space()

            exp.learn(wandb_config={"obs": obs})

            exp.finish_wandb()
