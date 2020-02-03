import numpy as np
import ignite
from ignite.engine import Events
from functools import partial

from .add_evaluation_logger import add_evaluation_logger
from .add_best_results_logger import add_best_results_logger
from workflow.ignite.write_engine_metrics import write_engine_metrics
from workflow.ignite.tqdm_print import tqdm_print
from workflow.ignite.constants import TQDM_OUTFILE


def add_default_event_handlers(
    model, optimizer, trainer, evaluator, validate_data_loader, score_function,
    config
):

    bar_format = '{desc} {percentage:3.0f}%|{bar}{postfix} {n}/{total} [{elapsed}<{remaining} {rate_fmt}]'

    # Order of attaching progress bars is important
    ignite.contrib.handlers.tqdm_logger.ProgressBar(
        desc='training', bar_format=bar_format, file=TQDM_OUTFILE
    ).attach(trainer)

    trainer.add_event_handler(
        Events.EPOCH_STARTED,
        lambda engine: tqdm_print(f'------ epoch: {engine.state.epoch} / {engine.state.max_epochs} ------')
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        partial(write_engine_metrics, name='training')
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda engine: evaluator.run(validate_data_loader)
    )

    ignite.contrib.handlers.tqdm_logger.ProgressBar(
        desc='validating', bar_format=bar_format, file=TQDM_OUTFILE
    ).attach(evaluator)

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        partial(write_engine_metrics, name='validating')
    )

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        ignite.handlers.ModelCheckpoint(
            dirname='checkpoints',
            filename_prefix='model',
            score_function=score_function,
            n_saved=1,
            require_empty=False,
        ),
        dict(
            model=model,
            optimizer=optimizer,
        ),
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan(),
    )

    early_stopping_handler = ignite.handlers.EarlyStopping(
        patience=config['patience'],
        score_function=score_function,
        trainer=trainer,
    )
    evaluator.add_event_handler(
        Events.COMPLETED, early_stopping_handler
    )

    # add_evaluation_logger(
    #     trainer, evaluator, validate_data_loader, early_stopping_handler
    # )

    add_best_results_logger(
        trainer, evaluator, score_function=score_function
    )
