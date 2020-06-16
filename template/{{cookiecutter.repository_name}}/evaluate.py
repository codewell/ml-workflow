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
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger, OutputHandler
)
import logging
import workflow
from workflow import json
from workflow.functional import starcompose
from workflow.torch import set_seeds
from workflow.ignite import worker_init
from workflow.ignite.handlers.learning_rate import (
    LearningRateScheduler, warmup, cyclical
)
from workflow.ignite.handlers import (
    EpochLogger,
    MetricsLogger,
    ProgressBar
)
from datastream import Datastream

from {{cookiecutter.package_name}} import data, architecture, metrics

logging.getLogger('ignite').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def loss(batch):
    return F.cross_entropy(
        batch['predicted_class_name'], batch['class_name'],
    )


def evaluate(config):
    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.SimpleModel().to(device)

    train_state = dict(model=model)

    print('Loading model checkpoint')
    workflow.ignite.handlers.ModelCheckpoint.load(
        train_state, 'model/checkpoints', device
    )


    @workflow.ignite.decorators.evaluate(model)
    def evaluate_batch(engine, batch):
        batch['predicted_class_name'] = model(batch['image'])
        batch['loss'] = loss(batch).item()
        return batch


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

    tensorboard_logger = TensorboardLogger(log_dir='tb')

    for evaluator_desc, data_loader in evaluate_data_loaders.items():
        engine = Engine(evaluate_batch)

        metrics_ = metrics()

        for metric_name, metric in metrics_.items():
            metric.attach(engine, metric_name)

        evaluator_metric_names = list(metrics_.keys())

        ProgressBar(desc=evaluator_desc).attach(engine)
        MetricsLogger(evaluator_desc).attach(engine, evaluator_metric_names)
        tensorboard_logger.attach(
            engine,
            OutputHandler(
                tag=evaluator_desc,
                metric_names=evaluator_metric_names,
            ),
            Events.EPOCH_COMPLETED,
        )

        engine.run(data=data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_batch_size', type=int, default=128)
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

