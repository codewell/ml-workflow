import ignite
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger, OutputHandler, OptimizerParamsHandler, global_step_from_engine
)

from workflow.ignite.handlers.epoch_logger import EpochLogger
from workflow.ignite.handlers.metrics_logger import MetricsLogger
from workflow.ignite.handlers.progress_bar import ProgressBar


PROGRESS_DESC = 'progress'
TRAIN_DESC = 'train'


def trainer(
    train_batch,
    evaluate_batch,
    evaluate_data_loaders,
    metrics,
    optimizers,
):
    '''
    Create standard trainer with evaluators.

    Parameters
    ----------
    train_batch : function
        function that trains on given batch
    evaluate_batch : function
        function that evaluates a given batch
    evaluate_data_loaders: list
        data loaders that yield batches to evaluate on
    metrics : dict
        dict with one dict each for 'train' and evaluate data loader. Wrap a
        metric with trainer.Progress to show in progress bar.
    optimizers : dict
        dict with optimizers for logging

    Returns
    -------
    tuple
        trainer engine
        list of evaluator engines
        tensorboard logger
    '''

    trainer = ignite.engine.Engine(train_batch)

    for name, metric in metrics.get(PROGRESS_DESC, dict()).items():
        metric.attach(trainer, name)

    for name, metric in metrics.get(TRAIN_DESC, dict()).items():
        metric.attach(trainer, name)

    evaluators = {
        evaluator_name: ignite.engine.Engine(evaluate_batch)
        for evaluator_name in evaluate_data_loaders.keys()
    }

    for evaluator_name, evaluator in evaluators.items():
        for metric_name, metric in metrics[evaluator_name].items():
            metric.attach(evaluator, metric_name)

    tensorboard_logger = TensorboardLogger(log_dir='tb')

    EpochLogger().attach(trainer)

    # Order of attaching progress bars is important for vscode / atom
    ProgressBar(desc=TRAIN_DESC).attach(
        trainer, metric_names=list(metrics.get(PROGRESS_DESC, dict()).keys())
    )
    tensorboard_logger.attach(
        trainer,
        OutputHandler(
            tag=PROGRESS_DESC,
            metric_names=list(metrics.get(PROGRESS_DESC, dict()).keys()),
        ),
        Events.ITERATION_COMPLETED,
    )

    MetricsLogger(TRAIN_DESC).attach(
        trainer, metrics.get(TRAIN_DESC, dict()).keys()
    )
    tensorboard_logger.attach(
        trainer,
        OutputHandler(
            tag=TRAIN_DESC,
            metric_names=list(metrics.get(TRAIN_DESC, dict()).keys()),
        ),
        Events.ITERATION_COMPLETED,
    )


    def run_evaluator(evaluator_desc):
        return lambda engine: evaluators[evaluator_desc].run(
            evaluate_data_loaders[evaluator_desc]
        )


    for evaluator_desc, evaluator in evaluators.items():
        evaluator_metric_names = list(metrics[evaluator_desc].keys())

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, run_evaluator(evaluator_desc),
        )

        ProgressBar(desc=evaluator_desc).attach(evaluator)
        MetricsLogger(evaluator_desc).attach(evaluator, evaluator_metric_names)
        tensorboard_logger.attach(
            evaluator,
            OutputHandler(
                tag=evaluator_desc,
                metric_names=evaluator_metric_names,
                global_step_transform=global_step_from_engine(trainer),
            ),
            Events.EPOCH_COMPLETED,
        )

    if type(optimizers) is not dict:
        optimizers = dict(optimizer=optimizers)

    for name, optimizer in optimizers.items():
        tensorboard_logger.attach(
            trainer,
            log_handler=OptimizerParamsHandler(
                tag=f'{TRAIN_DESC}/{name}',
                param_name='lr',
                optimizer=optimizer,
            ),
            event_name=Events.ITERATION_COMPLETED,
        )

    return trainer, evaluators, tensorboard_logger
