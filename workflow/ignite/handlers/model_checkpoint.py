import ignite
from ignite.engine import Events


class ModelCheckpoint:
    def __init__(self, model_score_function):
        self.model_checkpoint = ignite.handlers.ModelCheckpoint(
            dirname='checkpoints',
            filename_prefix='model',
            score_function=model_score_function,
            n_saved=1,
            require_empty=False,
        )

    def attach(self, engine, *args, **kwargs):
        engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.model_checkpoint,
            *args,
            **kwargs,
        )