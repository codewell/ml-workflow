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

from {{cookiecutter.package_name}} import (
    datastream, architecture, metrics, log_examples
)


def train(config):

    set_seeds(config['seed'])

    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.Model().to(device)
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

    def process_batch(examples):
        predictions = model.predictions(
            architecture.FeatureBatch.from_examples(examples)
        )
        loss = predictions.loss(examples)
        return predictions, loss

    @workflow.ignite.decorators.train(model, optimizer)
    def train_batch(engine, examples):
        predictions, loss = process_batch(examples)
        loss.backward()
        return dict(
            examples=examples,
            predictions=predictions.cpu().detach(),
            loss=loss,
        )

    @workflow.ignite.decorators.evaluate(model)
    def evaluate_batch(engine, examples):
        predictions, loss = process_batch(examples)
        return dict(
            examples=examples,
            predictions=predictions.cpu().detach(),
            loss=loss,
        )

    evaluate_data_loaders = {
        f'evaluate_{name}': datastream.data_loader(
            batch_size=config['eval_batch_size'],
            num_workers=config['n_workers'],
            collate_fn=tuple,
        )
        for name, datastream in datastream.evaluate_datastreams().items()
    }

    trainer, evaluators, tensorboard_logger = workflow.ignite.trainer(
        train_batch,
        evaluate_batch,
        evaluate_data_loaders,
        metrics=dict(
            progress=metrics.progress_metrics(),
            train=metrics.train_metrics(),
            **{
                name: metrics.evaluate_metrics()
                for name in evaluate_data_loaders.keys()
            }
        ),
        optimizers=optimizer,
    )

    workflow.ignite.handlers.ModelScore(
        lambda: -evaluators['evaluate_early_stopping'].state.metrics['loss'],
        train_state,
        {
            name: metrics.evaluate_metrics()
            for name in evaluate_data_loaders.keys()
        },
        tensorboard_logger,
        config,
    ).attach(trainer, evaluators)
    
    tensorboard_logger.attach(
        trainer,
        log_examples('train', trainer),
        ignite.engine.Events.EPOCH_COMPLETED,
    )
    tensorboard_logger.attach(
        evaluators['evaluate_compare'],
        log_examples('evaluate_compare', trainer),
        ignite.engine.Events.EPOCH_COMPLETED,
    )

    if config.get('search_learning_rate', False):

        def search(config):
            def search_(step, multiplier):
                return (
                    step,
                    (1 / config['minimum_learning_rate'])
                    ** (step / config['n_batches'])
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

    trainer.run(
        data=(
            datastream.GradientDatastream()
            .data_loader(
                batch_size=config['batch_size'],
                num_workers=config['n_workers'],
                n_batches_per_epoch=config['n_batches_per_epoch'],
                worker_init_fn=partial(worker_init, config['seed'], trainer),
                collate_fn=tuple,
            )
        ),
        max_epochs=config['max_epochs'],
    )
