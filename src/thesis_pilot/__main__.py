# -*- coding: utf-8 -*-
import click
import os
from experiments.core import Experiment
from collections import OrderedDict


@click.group(context_settings={'show_default': True})
@click.option('--no-log', '-nl', default=True, is_flag=True, help='Enable logging.')
@click.pass_context
def main(ctx, no_log):
    """Main CLI entrypoint."""
    ctx.ensure_object(dict)

    ctx.obj['no_log'] = no_log



class ConfigCommands(click.core.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.insert(0, click.core.Option(('--algo', '-a'), default="SAC", help='Algorithm to use.'))
        self.params.insert(0, click.core.Option(('--env', '-e'), default="citation", help='Environment to use.'))
        self.params.insert(0, click.core.Option(('--task', '-t'), default="aoa_sin", help='Task to use.'))
        self.params.insert(0, click.core.Option(('--seed', '-s'), default=None, help='Seed to use.'))
        self.params.insert(0, click.core.Option(('--dt', '-dt'), default=0.1, help='Time step to use.'))
        self.params.insert(0, click.core.Option(('--episode-steps', '-es'), default=100, help='Number of steps to use.'))
        self.params.insert(0, click.core.Option(('--learning-steps', '-ls'), default=1000, help='Number of learning steps to use.'))
        self.params.insert(0, click.core.Option(('--offline', '-o'), default=False, is_flag=True, help='Whether to run offline.'))
        self.params.insert(0, click.core.Option(('--project-name', '-pn'), default="", help='Name of the project.'))
        self.params.insert(0, click.core.Option(('--name', '-n'), default="", help='Name of the run.'))
        self.params.insert(0, click.core.Option(('--verbose', '-v'), default=2, help='Verbosity level.'))
        self.params.insert(0, click.core.Option(('--config', '-c'), default="sp", help='Configuration to use.'))
        self.params.insert(0, click.core.Option(('--run', '-r'), default=1, help='Number of runs to use.'))
        self.params.insert(0, click.core.Option(('--tags', '-tg'), default=None, help='Tags to use.'))

@main.command(cls=ConfigCommands)
@click.pass_context
@click.option('--exp', '-e', default=None, help='Name of the experiment to run.')
@click.option('--number', '-n', default=None, help='Number of the experiment to run.')
def experiment(ctx, exp: str, number: int, **kwargs):
    """Run an experiment."""
    from experiments import exp_best_algo

    exp_dict = OrderedDict()
    exp_dict["best_algo"] = exp_best_algo.main

    if exp is not None:
        exp_name = exp
    elif number is not None:
        exp_name = list(exp_dict.items())[int(number)][0]
    else:
        for i, exp_name in enumerate(exp_dict):
            print(f"{i}: {exp_name}")
        exp_name = None

    if exp_name == 'best_algo':
        exp_dict[exp_name](
            episode_steps=kwargs['episode_steps'],
            learning_steps=kwargs['learning_steps'],
            run=kwargs['run']
        )

@main.command(cls=ConfigCommands)
@click.pass_context
def learn(ctx, **kwargs):
    """Learn a task."""
    exp = Experiment(algorithm_name=kwargs['algo'],
                     env_name=kwargs['env'],
                     task_name=kwargs['task'],
                     seed=kwargs['seed'],
                     dt=kwargs['dt'],
                     episode_steps=kwargs['episode_steps'],
                     learning_steps=kwargs['learning_steps'],
                     verbose=kwargs['verbose'],
                     offline=kwargs['offline'],
                     configuration=kwargs['config'],
                     project_name=kwargs['project_name'],
                     run=kwargs['run'],
                     )

    exp.learn(name=kwargs['name'], tags=kwargs['tags'])


if __name__ == '__main__':
    main()
