from ignite.engine import Events

from workflow.ignite.tqdm_print import tqdm_print


class EpochLogger:
    def attach(self, engine):
        engine.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: tqdm_print(f'------ epoch: {engine.state.epoch} / {engine.state.max_epochs} ------')
        )
