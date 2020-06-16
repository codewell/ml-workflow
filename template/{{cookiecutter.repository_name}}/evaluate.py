import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from functools import partial
import numpy as np
import random
import argparse
import torch
import torch.nn.functional as F
import ignite
import logging
import workflow
from workflow import json
from workflow.functional import starcompose
from workflow.torch import set_seeds
from workflow.ignite import worker_init
from workflow.ignite.handlers.learning_rate import (
    LearningRateScheduler, warmup, cyclical
)
from datastream import Datastream

from {{cookiecutter.package_name}} import data, architecture

logging.getLogger('ignite').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate(config):
    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.SimpleModel().to(device)

    train_state = dict(model=model)

    model_checkpoint_handler = workflow.ignite.handlers.ModelCheckpoint()

    print('Loading model checkpoint')
    model_checkpoint_handler.load(
        train_state, f'model/{model_checkpoint_handler.dirname}', device
    )


    def loss(batch):
        return F.cross_entropy(
            batch['predicted_class_name'], batch['class_name'],
        )


    @workflow.ignite.decorators.evaluate(model)
    def evaluate_batch(engine, batch):
        batch['predicted_class_name'] = model(batch['image'])
        batch['loss'] = loss(batch).item()
        return batch


    def metrics():
        return dict(
            loss=ignite.metrics.Average(
                lambda output: output['loss']
            ),
        )


    evaluate_data_loaders = {
        f'evaluate_{name}': (
            Datastream(dataset)
            .map(architecture.preprocess)
            .data_loader(
                batch_size=config['eval_batch_size'],
                num_workers=config['n_workers'],
            )
        )
        for name, dataset in data.datasets().items()
    }

    ignite.Engine(evaluate_batch).run(
        data=(
            data.GradientDatastream()
            .map(architecture.preprocess)
            .data_loader(
                batch_size=config['batch_size'],
                num_workers=config['n_workers'],
                n_batches_per_epoch=config['n_batches_per_epoch'],
                worker_init_fn=partial(worker_init, config['seed'], trainer_),
            )
        ),
        max_epochs=config['max_epochs'],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--n_batches_per_epoch', default=50, type=int)
    parser.add_argument('--n_batches_per_step', default=1, type=int)
    parser.add_argument('--patience', type=float, default=10)
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
    )

    json.write(config, 'config.json')

    evaluate(config)

