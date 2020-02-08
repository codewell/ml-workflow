from tqdm import tqdm
from ignite.engine import Events


class EpochLogger:
    def attach(self, engine):
        engine.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: tqdm.write(f'------ epoch: {engine.state.epoch} / {engine.state.max_epochs} ------')
        )
