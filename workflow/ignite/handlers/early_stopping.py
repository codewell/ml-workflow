import ignite
from ignite.engine import Events

from workflow.ignite.tqdm_print import tqdm_print


class EarlyStopping:
    def __init__(self, model_score_function, trainer, config):
        self.early_stopping_handler = ignite.handlers.EarlyStopping(
            patience=config['patience'],
            score_function=model_score_function,
            trainer=trainer,
        )

    def attach(self, engine):
        engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.early_stopping_handler
        )

        engine.add_event_handler(
            Events.EPOCH_COMPLETED,
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
