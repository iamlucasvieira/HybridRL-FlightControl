"""Module that defines helper functions for wandb."""
import numpy as np
import torch
import wandb

from envs import BaseEnv, CitationEnv


def evaluate(agent, env, n_times=1, to_wandb=True, task=None):
    """Run the experiment n times."""

    for _ in range(n_times):
        if task is not None:
            env.set_task(task)
        obs, _ = env.reset()
        done = False
        steps = 0
        episode_return = 0
        while not done:
            action = agent.predict(obs, deterministic=True)

            # Transform action into numpy if it is a tensor
            if isinstance(action, torch.Tensor):
                action = action.detach().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward

            env.render()

            if wandb.run is not None and to_wandb:
                wandb.log({"reward": reward, "episode_step": steps})
                wandb.log(
                    {
                        "action": action,
                        "episode_step": steps,
                    }
                )

                if isinstance(env, BaseEnv):
                    log_base_env(env, steps, done)

                if isinstance(env, CitationEnv):
                    log_citation_env(env, steps)

            steps += 1
        print(f"finished at {steps}")
        return episode_return


def log_base_env(env: BaseEnv, step: int, done: bool):
    """Log the base environment information after a step."""

    for idx, tracked_state in enumerate(env.task.tracked_states):
        wandb.log(
            {
                f"eval/{tracked_state}_ref": env.reference[-1][idx],
                f"eval/{tracked_state}": env.track[-1][idx],
                "eval/step": step,
            }
        )
        wandb.log(
            {
                f"eval/{tracked_state}_sq_error": env.sq_error[-1][idx],
                "eval/step": step,
            }
        )

    for action_name, action_value in zip(env.input_names, env.actions[-1]):
        wandb.log(
            {
                f"eval/{action_name}": action_value,
                "eval/step": step,
            }
        )
    if done:
        wandb.log({"eval/nmae": env.nmae})


def log_citation_env(env: CitationEnv, step: int):
    """Log the Citation env information after a step."""
    for input_name, input_value in zip(env.input_names, env.actions[-1]):
        wandb.log(
            {
                f"citation_inputs/{input_name}": input_value,
                "citation_inputs/step": step,
            }
        )

    # Log states
    for state_name, state_value in zip(env.model.states, env.states[-1]):
        wandb.log(
            {
                f"citation_states/{state_name}": state_value,
                "citation_states/step": step,
            }
        )
