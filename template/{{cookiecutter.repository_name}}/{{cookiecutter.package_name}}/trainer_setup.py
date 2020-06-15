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


def trainer_setup(config):

    set_seeds(config['seed'])

    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.SimpleModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate']
    )

    train_state = dict(model=model, optimizer=optimizer)

    model_checkpoint_handler = workflow.ignite.handlers.ModelCheckpoint()

    if os.path.exists('model'):
        print('Loading model checkpoint')
        model_checkpoint_handler.load(
            train_state, f'model/{model_checkpoint_handler.dirname}', device
        )

        workflow.torch.set_learning_rate(optimizer, config['learning_rate'])

    n_parameters = sum([
        p.shape.numel() for p in model.parameters() if p.requires_grad
    ])
    print(f'n_parameters: {n_parameters:,}')


    def loss(batch):
        return F.cross_entropy(
            batch['predicted_class_name'], batch['class_name'],
        )


    @workflow.ignite.decorators.train(model, optimizer)
    def train_batch(engine, batch):
        batch['predicted_class_name'] = model(batch['image'])
        loss_ = loss(batch)
        loss_.backward()
        batch['loss'] = loss_.item()
        return batch


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


    train_metrics = dict(
        loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: output['loss'],
            epoch_bound=False,
        ),
    )

    progress_metrics = dict(
        batch_loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: output['loss'],
            epoch_bound=False,
            alpha=1e-7,
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

    trainer, evaluators, tensorboard_logger = workflow.ignite.trainer(
        train_batch,
        evaluate_batch,
        evaluate_data_loaders,
        metrics=dict(
            progress=progress_metrics,
            train=train_metrics,
            **{
                name: metrics()
                for name in evaluate_data_loaders.keys()
            }
        ),
        optimizers=optimizer,
    )

    workflow.ignite.handlers.ModelScore(
        lambda: evaluators['evaluate_early_stopping'].state.metrics['loss'],
        train_state,
        {
            name: metrics()
            for name in evaluate_data_loaders.keys()
        },
        tensorboard_logger,
        config,
    ).attach(trainer, evaluators)

    return dict(
        trainer=trainer,
        evaluators=evaluators,
        tensorboard_logger=tensorboard_logger,
        optimizer=optimizer,
    )
