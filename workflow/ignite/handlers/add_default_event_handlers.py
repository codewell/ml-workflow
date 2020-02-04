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
    trainer_metric_names,
    validator_metric_names,
    config,
):

    EpochLogger().attach(trainer)

    # Order of attaching progress bars is important for vscode / atom
    ProgressBar(desc='training').attach(
        trainer, trainer_metric_names
    )
    MetricsLogger('training').attach(trainer)
    TensorboardLogger(log_dir='tb').attach(
        trainer,
        OutputHandler(
            tag='training',
            metric_names=trainer_metric_names,
        ),
        Events.ITERATION_COMPLETED,
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda engine: validator.run(validate_data_loader)
    )

    validator_desc = 'validating'
    ProgressBar(desc=validator_desc).attach(validator)
    MetricsLogger(validator_desc).attach(validator)
    TensorboardLogger(log_dir='tb').attach(
        validator,
        OutputHandler(
            tag=validator_desc,
            metric_names=validator_metric_names,
            global_step_transform=global_step_from_engine(trainer),
        ),
        Events.EPOCH_COMPLETED,
    )

    ModelCheckpoint(model_score_function).attach(validator, dict(
        model=model,
        optimizer=optimizer,
    ))

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan(),
    )

    EarlyStopping(model_score_function, trainer, config).attach(validator)
