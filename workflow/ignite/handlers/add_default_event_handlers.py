import ignite
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger, OutputHandler, global_step_from_engine
)

from workflow.ignite.tqdm_print import tqdm_print
from .early_stopping import EarlyStopping
from .epoch_logger import EpochLogger
from .metrics_logger import MetricsLogger
from .model_checkpoint import ModelCheckpoint
from .progress_bar import ProgressBar


def add_default_event_handlers(
    model,
    optimizer,
    trainer,
    validator,
    validate_data_loader,
    model_score_function,
    trainer_metrics,
    validator_metrics,
    config,
):

    for name, metric in trainer_metrics.items():
        metric.attach(trainer, name)

    for name, metric in validator_metrics.items():
        metric.attach(validator, name)


    EpochLogger().attach(trainer)

    # Order of attaching progress bars is important for vscode / atom
    training_desc = 'train'
    ProgressBar(desc=training_desc).attach(
        trainer, 'all'
    )
    MetricsLogger(training_desc).attach(trainer)
    TensorboardLogger(log_dir='tb').attach(
        trainer,
        OutputHandler(
            tag=training_desc,
            metric_names='all',
        ),
        Events.ITERATION_COMPLETED,
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda engine: validator.run(validate_data_loader)
    )

    validator_desc = 'validate'
    ProgressBar(desc=validator_desc).attach(validator)
    MetricsLogger(validator_desc).attach(validator)
    TensorboardLogger(log_dir='tb').attach(
        validator,
        OutputHandler(
            tag=validator_desc,
            metric_names='all',
            global_step_transform=global_step_from_engine(trainer),
        ),
        Events.EPOCH_COMPLETED,
    )

    ModelCheckpoint(model_score_function).attach(
        validator,
        dict(
            model=model,
            optimizer=optimizer,
        )
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan(),
    )

    EarlyStopping(model_score_function, trainer, config).attach(validator)
