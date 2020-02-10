from tqdm import tqdm
from ignite.engine import Events


def get_epoch_text(epoch, max_epochs):
    return f'------ epoch: {epoch} / {max_epochs} ------'


class EpochLogger:
    def attach(self, engine):
        engine.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: tqdm.write(
                get_epoch_text(engine.state.epoch, engine.state.max_epochs)
            )
        )
