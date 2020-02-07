import ignite
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger, OutputHandler, OptimizerParamsHandler, global_step_from_engine
)

from workflow.ignite.tqdm_print import tqdm_print
from workflow.ignite.handlers.early_stopping import EarlyStopping
from workflow.ignite.handlers.epoch_logger import EpochLogger
from workflow.ignite.handlers.metrics_logger import MetricsLogger
from workflow.ignite.handlers.model_checkpoint import ModelCheckpoint
from workflow.ignite.handlers.progress_bar import ProgressBar


def create_standard_trainer_validator(
    model,
    optimizer,
    train_batch,
    evaluate_batch,
    validate_data_loader,
    model_score_function,
    trainer_metrics,
    validator_metrics,
    config,
):

    trainer = ignite.engine.Engine(train_batch)
    validator = ignite.engine.Engine(evaluate_batch)

    for name, metric in trainer_metrics.items():
        metric.attach(trainer, name)

    for name, metric in validator_metrics.items():
        metric.attach(validator, name)

    tensorboard_logger = TensorboardLogger(log_dir='tb')

    EpochLogger().attach(trainer)

    # Order of attaching progress bars is important for vscode / atom
    training_desc = 'train'
    ProgressBar(desc=training_desc).attach(
        trainer, 'all'
    )
    MetricsLogger(training_desc).attach(trainer)
    tensorboard_logger.attach(
        trainer,
        OutputHandler(
            tag=training_desc,
            metric_names='all',
        ),
        Events.ITERATION_COMPLETED,
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda engine: validator.run(validate_data_loader),
    )

    validator_desc = 'validate'
    ProgressBar(desc=validator_desc).attach(validator)
    MetricsLogger(validator_desc).attach(validator)
    tensorboard_logger.attach(
        validator,
        OutputHandler(
            tag=validator_desc,
            metric_names='all',
            global_step_transform=global_step_from_engine(trainer),
        ),
        Events.EPOCH_COMPLETED,
    )

    tensorboard_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(
            tag=training_desc,
            param_name='lr',
            optimizer=optimizer,
        ),
        event_name=Events.ITERATION_COMPLETED,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan(),
    )

    _model_score_function = lambda trainer: (
        model_score_function(validator)
    )

    ModelCheckpoint(_model_score_function).attach(
        trainer,
        dict(
            model=model,
            optimizer=optimizer,
        )
    )

    EarlyStopping(_model_score_function, trainer, config).attach(trainer)

    return trainer, validator
