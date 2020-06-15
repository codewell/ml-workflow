import argparse
import os
import logging
from functools import partial
import torch
from workflow import json
from workflow.functional import starcompose
from workflow.ignite import worker_init
from workflow.ignite.handlers.learning_rate import (
    LearningRateScheduler, warmup, cyclical
)

from {{cookiecutter.package_name}} import data, architecture, trainer_setup

logging.getLogger('ignite').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def search(config):
    def search_(step, multiplier):
        return (
            step,
            (1 / config['learning_rate']) ** (  step / (
                config['n_batches']
            ))
        )
    return search_


def search_learning_rate(config):
    trainer_setup_ = trainer_setup(config)
    trainer_ = trainer_setup_['trainer']

    LearningRateScheduler(
        trainer_setup_['optimizer'],
        search(config),
    ).attach(trainer_)

    trainer_.run(
        data=(
            data.GradientDatastream()
            .map(architecture.preprocess)
            .data_loader(
                batch_size=config['batch_size'],
                num_workers=config['n_workers'],
                n_batches_per_epoch=config['n_batches'],
                worker_init_fn=partial(worker_init, config['seed'], trainer_),
            )
        ),
        max_epochs=1,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--minimum_learning_rate', type=float, default=1e-7)
    parser.add_argument('--n_batches', default=200, type=int)
    parser.add_argument('--n_batches_per_step', default=1, type=int)
    parser.add_argument('--patience', type=float, default=1)
    parser.add_argument('--n_workers', default=2, type=int)

    try:
        __IPYTHON__
        args = parser.parse_known_args()[0]
    except NameError:
        args = parser.parse_args()

    config = vars(args)
    config.update(
        seed=1,
        use_cuda=torch.cuda.is_available(),
        run_id=os.getenv('RUN_ID'),
        learning_rate=config['minimum_learning_rate'],
    )

    json.write(config, 'config.json')

    search_learning_rate(config)
