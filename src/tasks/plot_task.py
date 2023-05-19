"""Script used to visulize the tasks."""

import matplotlib.pyplot as plt
import numpy as np

from envs.citation.citation_env import CitationEnv
from tasks import get_task


episode_steps = 2_000
task_name = "att_train"

env = CitationEnv(episode_steps=episode_steps)
x = np.arange(0, episode_steps * env.dt, env.dt)
task = get_task(task_name)(env)

references = []
for i in range(episode_steps):
    references.append(task.reference())
    env.current_time += env.dt
references = np.array(references)

n_states = references.shape[1]

# Create figure with the number of states in each frame
fig, axs = plt.subplots(n_states, 1, figsize=(10, 10))
fig.suptitle(f"Task: {task_name}")

for idx, name in enumerate(task.tracked_states):
    axs[idx].plot(x, references[:, idx], label="Reference")
    axs[idx].set_ylabel(f"{name}")
    axs[idx].legend()

plt.show()

# Create 3D plot
if n_states == 3:
    ax = plt.axes(projection="3d")
    ax.plot3D(references[:, 0], references[:, 1], references[:, 2], "gray")
    x_label, y_label, z_label = task.tracked_states
    states_dict = {
        x_label: references[:, 0],
        y_label: references[:, 1],
        z_label: references[:, 2],
    }

    ax.plot3D(states_dict["beta"], states_dict["phi"], states_dict["theta"], "gray")
    ax.set_xlabel("beta")
    ax.set_ylabel("phi")
    ax.set_zlabel("theta")

plt.show()
