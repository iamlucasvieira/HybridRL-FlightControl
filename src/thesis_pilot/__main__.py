# -*- coding: utf-8 -*-
import click
import os
from experiments.core import Experiment


@click.group(context_settings={'show_default': True})
@click.option('--no-log', '-nl', default=True, is_flag=True, help='Enable logging.')
@click.pass_context
def main(ctx, no_log):
    """Main CLI entrypoint."""
    ctx.ensure_object(dict)

    ctx.obj['no_log'] = no_log


@main.command()
@click.option('--algo', '-a', default='SAC', help='Algorithm to use.')
@click.option('--env', '-e', default='citation', help='Environment to use.')
@click.option('--task', '-t', default='aoa', help='Task to use.')
@click.option('--seed', '-s', default=1, help='Random seed.')
@click.option('--dt', '-dt', default=0.1, help='Time step.')
@click.option('--episode-steps', '-es', default=100, help='Number of steps in an episode.')
@click.option('--global-steps', '-gs', default=1_000, help='Number of total learning steps.')
@click.option('--offline', '-off', default=False, is_flag=True, help='Disable wandb sync.')
@click.option('--verbose', '-v', default=1, help='Verbosity level.')
@click.option("--plot", "-p", default=False, is_flag=True, help="Plot results.")
@click.option("--name", "-n", default=None, help="Name of the run.")
@click.option("--tags", "-tg", default=None, help="Tags of the run.")
@click.option("--config", "-c", default="symmetric", help="Configuration of the aircraft.")
@click.option("--project", "-pr", default="thesis-pilot", help="Name of the project.")
@click.option("--run", "-r", default=1, help="Number of times to run the environment after learning.")
@click.pass_context
def learn(ctx, algo: str, env: str, task: str, seed: int, dt: float, episode_steps: int, global_steps: int,
          offline: bool, verbose: int, plot: bool, name: str, tags: str, config: str, project: str, run: int):
    exp = Experiment(algorithm_name=algo,
                     env_name=env,
                     task_name=task,
                     seed=seed,
                     dt=dt,
                     episode_steps=episode_steps,
                     learning_steps=global_steps,
                     verbose=verbose,
                     offline=offline,
                     configuration=config,
                     project_name=project,
                     run=run,
                     )

    exp.learn(name=name, tags=tags)


if __name__ == '__main__':
    main()
