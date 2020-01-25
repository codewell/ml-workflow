from tqdm import tqdm
from ignite.engine import Events
from workflow.ignite.constants import TQDM_OUTFILE


def attach_train_progress_bar(trainer, config):

    epoch_progress_bar = tqdm(
        total=config['steps_per_epoch'],
        file=TQDM_OUTFILE,
        desc=f'Epoch [0 / {config["max_epochs"]}]'
    )

    @trainer.on(Events.EPOCH_STARTED)
    def reset_epoch_progress_bar(trainer):
        epoch_progress_bar.reset()

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_epoch_progress_bar(trainer):
        epoch_progress_bar.update()
        epoch_progress_bar.set_description(
            f'Epoch [{trainer.state.epoch} / {config["max_epochs"]}]'
            f' Loss: {trainer.state.metrics["running avg loss"]:.4f}'
        )
