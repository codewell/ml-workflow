import numpy as np
import ignite
from ignite.engine import Events
from .add_train_progress_bar import add_train_progress_bar
from .add_validation_progress_bar import add_validation_progress_bar
from .add_evaluation_logger import add_evaluation_logger
from .add_best_results_logger import add_best_results_logger


# def loss_score_function(engine):
#     score = -engine.state.metrics['loss']
#     return -np.inf if score == 0 else score


def add_default_event_handlers(
    model, optimizer, trainer, evaluator, validate_data_loader, score_function,
    config
):
    add_train_progress_bar(trainer, config)
    add_validation_progress_bar(evaluator, len(validate_data_loader), config)

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        ignite.handlers.ModelCheckpoint(
            dirname='checkpoints',
            filename_prefix='model',
            score_function=score_function,
            n_saved=1,
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

    add_evaluation_logger(
        trainer, evaluator, validate_data_loader, early_stopping_handler
    )

    add_best_results_logger(
        trainer, evaluator, score_function=score_function
    )
