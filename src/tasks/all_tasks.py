from tasks.attitude import AttitudeTrain, SineAttitude
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
    "sin_att": SineAttitude,
}
