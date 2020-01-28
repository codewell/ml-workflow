import ignite
from ignite.engine import Events
from tqdm import tqdm

from workflow.ignite.constants import TQDM_OUTFILE


def attach_train_progress_bar(
    trainer, config, output_transform=lambda batch: batch['loss']
):

    ignite.metrics.RunningAverage(
        output_transform=output_transform, alpha=0.98
    ).attach(trainer, 'running_average_loss')

    epoch_progress_bar = tqdm(
        total=config['batches_per_epoch'],
        file=TQDM_OUTFILE,
        desc=f'''Epoch [0 / {config['max_epochs']}]'''
    )

    @trainer.on(Events.EPOCH_STARTED)
    def reset_epoch_progress_bar(trainer):
        epoch_progress_bar.reset()

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_epoch_progress_bar(trainer):
        epoch_progress_bar.update()
        epoch_progress_bar.set_description(' '.join([
            f'''Epoch [{trainer.state.epoch} / {config['max_epochs']}]''',
            f'''Loss: {trainer.state.metrics['running_average_loss']:.4f}''',
        ]))
