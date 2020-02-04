import numpy as np
import ignite
from ignite.engine import Events
from functools import partial

from .add_best_results_logger import add_best_results_logger
from workflow.ignite.tqdm_print import tqdm_print
from workflow.ignite.constants import TQDM_OUTFILE
from .early_stopping import EarlyStopping
from .epoch_logger import EpochLogger
from .metrics_logger import MetricsLogger
from .model_checkpoint import ModelCheckpoint
from .progress_bar import ProgressBar


def add_default_event_handlers(
    model, optimizer, trainer, evaluators, validate_data_loader, score_function,
    config
):

    if type(evaluators) != dict:
        evaluators = dict(validating=evaluators)

    EpochLogger().attach(trainer)

    # Order of attaching progress bars is important
    ProgressBar(desc='training').attach(
        trainer,
        output_transform=lambda output: dict(loss=output['loss']),
        # ['running_loss'],
    )
    MetricsLogger('training').attach(trainer)

    for evaluator_desc, evaluator in evaluators.items():
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda engine: evaluator.run(validate_data_loader)
        )

        ProgressBar(desc=evaluator_desc).attach(evaluator)
        MetricsLogger(evaluator_desc).attach(evaluator)

    # ignite.contrib.handlers.tensorboard_logger.TensorboardLogger(
    #     log_dir='tb'
    # ).attach(
    #     trainer,
    #     ignite.contrib.handlers.tensorboard_logger.OutputHandler(
    #         tag='training',
    #         output_transform=lambda output: dict(loss=output['loss']),
    #     ),
    #     Events.ITERATION_COMPLETED,
    # )

    ModelCheckpoint(score_function).attach(evaluator, dict(
        model=model,
        optimizer=optimizer,
    ))

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan(),
    )

    EarlyStopping(score_function, trainer, config).attach(evaluator)


    # add_best_results_logger(
    #     trainer, evaluator, score_function=score_function
    # )
