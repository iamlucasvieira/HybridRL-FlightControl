from tasks.attitude import (
    AttitudeEval,
    AttitudeTrain,
    SinAttitudeEvaluate,
    SineAttitude,
)
from tasks.experiment_1 import CossStep, FixedSineAttitude, Hold, PseudoRandomSine
from tasks.sinusoidal import SineQ, SineTheta


def get_task(task_type: str):
    """Returns the task function."""
    if task_type in AVAILABLE_TASKS:
        return AVAILABLE_TASKS[task_type]
    else:
        raise ValueError(f"Task type {task_type} is not available.")


AVAILABLE_TASKS = {
    "sin_q": SineQ,
    "sin_theta": SineTheta,
    "att_train": AttitudeTrain,
    "att_eval": AttitudeEval,
    "sin_att": SineAttitude,
    "sin_att_eval": SinAttitudeEvaluate,
    "exp1_fixed_sin": FixedSineAttitude,
    "exp1_pseudo_random_sin": PseudoRandomSine,
    "exp1_coss_step": CossStep,
    "exp1_hold": Hold,
}
