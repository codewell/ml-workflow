from ignite.engine import Engine, Events
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger, OutputHandler
)

from workflow.ignite.handlers import (
    MetricsLogger,
    ProgressBar,
)


def evaluator(evaluate_batch, description, metrics, tensorboard_logger):
    engine = Engine(evaluate_batch)

    for metric_name, metric in metrics.items():
        metric.attach(engine, metric_name)

    metric_names = list(metrics.keys())

    ProgressBar(desc=description).attach(engine)
    MetricsLogger(description).attach(engine, metric_names)
    tensorboard_logger.attach(
        engine,
        OutputHandler(
            tag=description,
            metric_names=metric_names,
        ),
        Events.EPOCH_COMPLETED,
    )

    return engine
