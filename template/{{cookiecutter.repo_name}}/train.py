import os
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
from workflow.ignite.handlers.learning_rate import (
    LearningRateScheduler, warmup, cyclical
)

from {{cookiecutter.package_name}} import data, architecture


logging.getLogger('ignite').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def set_seeds(seed):
    np.random.seed(seed=seed + 1)
    random.seed(seed + 2)
    torch.manual_seed(seed + 3)


def train_model(config):

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
            workflow.torch.Datastream(dataset)
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

    model_checkpoint_handler.attach(trainer, train_state)

    LearningRateScheduler(
        optimizer,
        starcompose(
            warmup(150),
            cyclical(length=500),
        ),
    ).attach(trainer)

    trainer.run(
        data=(
            data.GradientDatastream()
            .map(architecture.preprocess)
            .data_loader(
                batch_size=config['batch_size'],
                num_workers=config['n_workers'],
                n_batches_per_epoch=config['n_batches_per_epoch'],
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
    parser.add_argument('--n_workers', default=2, type=int)
    args = parser.parse_args()

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

    set_seeds(config['seed'])

    json.write(config, 'config.json')

    train_model(config)
