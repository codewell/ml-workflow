import ignite
from ignite.engine import Events
from tqdm import tqdm

from workflow.ignite.constants import TQDM_OUTFILE


def add_train_progress_bar(
    trainer, config, output_transform=lambda batch: batch['loss']
):

    ignite.metrics.RunningAverage(
        output_transform=output_transform, alpha=0.98
    ).attach(trainer, 'running_average_loss')

    epoch = 1

    def create_progress_bar():
        return tqdm(
            total=config['n_batches_per_epoch'],
            file=TQDM_OUTFILE,
            desc=f'''Epoch [{epoch} / {config['max_epochs']}]''',
            leave=False
        )

    epoch_progress_bar = None

    @trainer.on(Events.EPOCH_STARTED)
    def reset_epoch_progress_bar(engine):
        nonlocal epoch_progress_bar
        epoch_progress_bar = create_progress_bar()

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_epoch_progress_bar(engine):
        epoch_progress_bar.update()
        epoch_progress_bar.set_description(' '.join([
            f'''Epoch [{engine.state.epoch} / {config['max_epochs']}]''',
            f'''Loss: {engine.state.metrics['running_average_loss']:.4f}''',
        ]))

    @trainer.on(Events.EPOCH_COMPLETED)
    @trainer.on(Events.EXCEPTION_RAISED)
    @trainer.on(Events.COMPLETED)
    def release_progress_bar(engine, *args):
        epoch_progress_bar.close()
