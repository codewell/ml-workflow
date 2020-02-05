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


def create_standard_trainer_evaluators(
    model,
    optimizer,
    train_batch,
    evaluate_batch,
    evaluate_data_loaders,
    model_score_function,
    trainer_metrics,
    create_evaluator_metrics,
    config,
):

    trainer = ignite.engine.Engine(train_batch)

    for name, metric in trainer_metrics.items():
        metric.attach(trainer, name)


    if type(evaluate_data_loaders) != dict:
        evaluate_data_loaders = dict(validate=evaluate_data_loaders)
        evaluators = dict(validate=ignite.engine.Engine(evaluate_batch))

        _model_score_function = lambda trainer: (
            model_score_function(evaluators['validate'])
        )
    else:
        evaluators = {
            evaluator_name: ignite.engine.Engine(evaluate_batch)
            for evaluator_name in evaluate_data_loaders.keys()
        }

        _model_score_function = lambda trainer: (
            model_score_function(evaluators)
        )

    for evaluator in evaluators.values():
        for metric_name, metric in create_evaluator_metrics().items():
            metric.attach(evaluator, metric_name)


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


    def run_evaluator(evaluator_desc):
        return lambda engine: evaluators[evaluator_desc].run(
            evaluate_data_loaders[evaluator_desc]
        )


    for evaluator_desc, evaluator in evaluators.items():

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, run_evaluator(evaluator_desc),
        )

        ProgressBar(desc=evaluator_desc).attach(evaluator)
        MetricsLogger(evaluator_desc).attach(evaluator)
        tensorboard_logger.attach(
            evaluator,
            OutputHandler(
                tag=evaluator_desc,
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
        event_name=Events.ITERATION_STARTED,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan(),
    )

    ModelCheckpoint(_model_score_function).attach(
        trainer,
        dict(
            model=model,
            optimizer=optimizer,
        )
    )

    EarlyStopping(_model_score_function, trainer, config).attach(trainer)

    return trainer, evaluators
