import numpy as np
import ignite
from ignite.engine import Events
from functools import partial
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger, OutputHandler, global_step_from_engine
)

from workflow.ignite.tqdm_print import tqdm_print
from workflow.ignite.constants import TQDM_OUTFILE
from .early_stopping import EarlyStopping
from .epoch_logger import EpochLogger
from .metrics_logger import MetricsLogger
from .model_checkpoint import ModelCheckpoint
from .progress_bar import ProgressBar


def add_default_event_handlers(
    model, optimizer, trainer, evaluators, validate_data_loader,
    model_score_function, tensorboard_metric_names, config
):

    if type(evaluators) != dict:
        evaluators = dict(validating=evaluators)

    EpochLogger().attach(trainer)

    # Order of attaching progress bars is important for vscode / atom
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

    ModelCheckpoint(model_score_function).attach(evaluator, dict(
        model=model,
        optimizer=optimizer,
    ))

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan(),
    )

    TensorboardLogger(log_dir='tb').attach(
        trainer,
        OutputHandler(
            tag='training',
            output_transform=lambda output: dict(loss=output['loss']),
        ),
        Events.ITERATION_COMPLETED,
    )

    for evaluator_desc, evaluator in evaluators.items():
        TensorboardLogger(log_dir='tb').attach(
            evaluator,
            OutputHandler(
                tag=evaluator_desc,
                metric_names=tensorboard_metric_names,
                global_step_transform=global_step_from_engine(trainer),
            ),
            Events.EPOCH_COMPLETED,
        )

    EarlyStopping(model_score_function, trainer, config).attach(evaluator)
