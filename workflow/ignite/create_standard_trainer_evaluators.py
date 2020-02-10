import ignite
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger, OutputHandler, OptimizerParamsHandler, global_step_from_engine
)
from workflow.ignite.metrics import ReduceMetricsLambda
from workflow.ignite.handlers import (
    EarlyStopping,
    EpochLogger,
    MetricsLogger,
    ModelCheckpoint,
    ProgressBar,
)


def create_standard_trainer_evaluators(
    model,
    optimizer,
    train_batch,
    evaluate_batch,
    evaluate_data_loaders,
    model_score_function,
    trainer_metrics,
    evaluator_metrics,
    config,
):

    trainer = ignite.engine.Engine(train_batch)

    for name, metric in trainer_metrics.items():
        metric.attach(trainer, name)

    evaluators = {
        evaluator_name: ignite.engine.Engine(evaluate_batch)
        for evaluator_name in evaluate_data_loaders.keys()
    }

    for evaluator_name, evaluator in evaluators.items():
        for metric_name, metric in evaluator_metrics[evaluator_name].items():
            metric.attach(evaluator, metric_name)

    tensorboard_logger = TensorboardLogger(log_dir='tb')

    EpochLogger().attach(trainer)

    progress_bar_metric_names = [
        name for name, metric in trainer_metrics.items()
        if trainer.has_event_handler(
            metric.iteration_completed, Events.ITERATION_COMPLETED
        )
    ]

    # Order of attaching progress bars is important for vscode / atom
    training_desc = 'train'
    ProgressBar(desc=training_desc).attach(
        trainer, progress_bar_metric_names
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
        event_name=Events.ITERATION_COMPLETED,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan(),
    )

    # ignite.metrics.MetricsLambda(
    #     lambda: model_score_function(evaluators)
    # ).attach(trainer, 'model_score')

    # ReduceMetricsLambda(
    #     max, lambda: model_score_function(evaluators)
    # ).attach(trainer, 'best_model_score')

    _model_score_function = lambda trainer: (
        model_score_function(evaluators)
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
