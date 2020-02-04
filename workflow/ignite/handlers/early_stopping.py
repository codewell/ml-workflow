import ignite
from ignite.engine import Events

from workflow.ignite.tqdm_print import tqdm_print


class EarlyStopping:
    def __init__(self, score_function, trainer, config):
        self.early_stopping_handler = ignite.handlers.EarlyStopping(
        patience=config['patience'],
        score_function=score_function,
        trainer=trainer,
    )

    def attach(self, engine):
        engine.add_event_handler(
            Events.COMPLETED, self.early_stopping_handler
        )

        engine.add_event_handler(
            Events.COMPLETED,
            self._print_status,
        )

    def _print_status(self, engine):
        epochs_until_stop = (
            self.early_stopping_handler.patience - self.early_stopping_handler.counter
        )
        tqdm_print(
            f'best score so far: {self.early_stopping_handler.best_score:.4f}'
            f' (stopping in {epochs_until_stop} epochs)\n'
        )
