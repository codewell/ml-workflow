import numpy as np
import ignite
from ignite.engine import Events
from functools import partial

from .add_best_results_logger import add_best_results_logger
from workflow.ignite.tqdm_print import tqdm_print
from workflow.ignite.constants import TQDM_OUTFILE
from .epoch_logger import EpochLogger
from .metrics_logger import MetricsLogger
from .model_checkpoint import ModelCheckpoint
from .progress_bar import ProgressBar


def add_default_event_handlers(
    model, optimizer, trainer, evaluators, validate_data_loader, score_function,
    config
):

    if type(evaluators) != list:
        evaluators = [evaluators]

    EpochLogger().attach(trainer)

    # Order of attaching progress bars is important
    ProgressBar(desc='training').attach(
        trainer,
        output_transform=lambda output: dict(loss=output['loss']),
        # ['running_loss'],
    )
    MetricsLogger('training').attach(trainer)

    for evaluator in evaluators:
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda engine: evaluator.run(validate_data_loader)
        )

        ProgressBar(desc='validating').attach(evaluator)
        MetricsLogger('validating').attach(evaluator)

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

    # trainer.add_event_handler(
    #     Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan(),
    # )

    # extend early stopping to be verbose?

    # early_stopping_handler = ignite.handlers.EarlyStopping(
    #     patience=config['patience'],
    #     score_function=score_function,
    #     trainer=trainer,
    # )
    # evaluator.add_event_handler(
    #     Events.COMPLETED, early_stopping_handler
    # )

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def print_early_stopping(engine):
    #     epochs_until_stop = (
    #         early_stopping_handler.patience - early_stopping_handler.counter
    #     )
    #     tqdm_print(
    #         f'best score so far: {early_stopping_handler.best_score:.4f}'
    #         f' (stopping in {epochs_until_stop} epochs)\n'
    #     )

    # add_best_results_logger(
    #     trainer, evaluator, score_function=score_function
    # )
