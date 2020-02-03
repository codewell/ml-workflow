from tqdm import tqdm
from ignite.engine import Events
from workflow.ignite.tqdm_print import tqdm_print
from workflow.ignite.write_engine_metrics import write_engine_metrics


def add_evaluation_logger(
        trainer, evaluator, validation_loader, early_stopping_handler=None
    ):

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_evaluation_results(trainer):
        tqdm_print(f'----- Epoch: {trainer.state.epoch} -----')
        write_engine_metrics(trainer, name='Training')
        write_engine_metrics(evaluator, name='Validation')
        if early_stopping_handler is not None:
            epochs_until_stop = (
                early_stopping_handler.patience - early_stopping_handler.counter
            )
            tqdm_print(
                f'Best score so far: {early_stopping_handler.best_score:.4f}'
                f' (stopping in {epochs_until_stop} epochs)\n'
            )
