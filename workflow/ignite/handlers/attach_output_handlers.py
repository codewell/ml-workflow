import ignite
from ignite.engine import Events
from .loss_score_function import loss_score_function
from .attach_validation_progress_bar import attach_validation_progress_bar
from .attach_evaluation_logger import attach_evaluation_logger
from .attach_best_results_logger import attach_best_results_logger


def attach_output_handlers(
        model, optimizer, trainer, evaluator, validation_loader, config,
        score_function=None
    ):

    if score_function is None:
        score_function = loss_score_function

    early_stopping_handler = ignite.handlers.EarlyStopping(
        patience=config['patience'],
        score_function=score_function,
        trainer=trainer,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        early_stopping_handler
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        ignite.handlers.TerminateOnNan(),
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        ignite.handlers.ModelCheckpoint(
            dirname='checkpoints',
            filename_prefix='model',
            score_function=score_function,
            n_saved=1,
        ),
        dict(
            weights=model,
            optimizer=optimizer,
        ),
    )
    attach_train_progress_bar(trainer, config)
    attach_validation_progress_bar(evaluator, validation_loader, config)
    attach_evaluation_logger(
        trainer, evaluator, validation_loader, early_stopping_handler
    )
    attach_best_results_logger(
        trainer, evaluator, score_function=score_function
    )
