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

from {{cookiecutter.package_name}} import data, architecture, metrics


def train(config):

    set_seeds(config['seed'])

    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.SimpleModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate']
    )

    train_state = dict(model=model, optimizer=optimizer)

    if os.path.exists('model'):
        print('Loading model checkpoint')
        workflow.ignite.handlers.ModelCheckpoint.load(
            train_state, 'model/checkpoints', device
        )

        workflow.torch.set_learning_rate(optimizer, config['learning_rate'])

    n_parameters = sum([
        p.shape.numel() for p in model.parameters() if p.requires_grad
    ])
    print(f'n_parameters: {n_parameters:,}')


    def loss(batch):
        return F.cross_entropy(
            batch['predicted_logits'], batch['class_index'],
        )


    @workflow.ignite.decorators.train(model, optimizer)
    def train_batch(engine, batch):
        batch['predicted_logits'] = model(batch['image'])
        loss_ = loss(batch)
        loss_.backward()
        batch['loss'] = loss_.item()
        return batch


    @workflow.ignite.decorators.evaluate(model)
    def evaluate_batch(engine, batch):
        batch['predicted_logits'] = model(batch['image'])
        batch['loss'] = loss(batch).item()
        return batch


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

    if config.get('search_learning_rate', False):

        def search(config):
            def search_(step, multiplier):
                return (
                    step,
                    (1 / config['minimum_learning_rate']) ** (  step / (
                        config['n_batches']
                    ))
                )
            return search_

        LearningRateScheduler(
            optimizer,
            search(config),
        ).attach(trainer)

    else:
        LearningRateScheduler(
            optimizer,
            starcompose(
                warmup(150),
                cyclical(length=500),
            ),
        ).attach(trainer)

    # Avoid ReproducibleBatchSampler. Should be fixed in ignite==0.4.0
    ignite.engine.engine.ReproducibleBatchSampler.__iter__ = (
        lambda self: iter(self.batch_sampler)
    )

    trainer.run(
        data=(
            data.GradientDatastream()
            .map(architecture.preprocess)
            .data_loader(
                batch_size=config['batch_size'],
                num_workers=config['n_workers'],
                n_batches_per_epoch=config['n_batches_per_epoch'],
                worker_init_fn=partial(worker_init, config['seed'], trainer),
            )
        ),
        max_epochs=config['max_epochs'],
    )
