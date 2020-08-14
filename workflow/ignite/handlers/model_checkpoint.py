import os
import torch
import numpy as np
import ignite
from ignite.engine import Events


class ModelCheckpoint:
    dirname = 'checkpoints'
    filename_prefix = 'model'

    def __init__(self, model_score_function=None):
        self.model_checkpoint = ignite.handlers.ModelCheckpoint(
            dirname=self.dirname,
            filename_prefix=self.filename_prefix,
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

    @staticmethod
    def load(
        to_load,
        dirname=None,
        device=None,
        suffix=None,
    ):
        if dirname is None:
            dirname = ModelCheckpoint.dirname

        models = os.listdir(dirname)
        if suffix is None:
            suffixes = [
                '_'.join(
                    os.path.splitext(name)[0]
                    .lstrip(ModelCheckpoint.filename_prefix)
                    .split('_')[2:]
                )
                for name in models
            ]
            suffix = suffixes[np.argmax([float(s.split('_')[-1]) for s in suffixes])]

        saved_checkpoint_state = torch.load(
            f'{dirname}/{ModelCheckpoint.filename_prefix}_checkpoint_{suffix}.pt',
            map_location=device,
        )

        for name, module_or_optimizer in to_load.items():
            module_or_optimizer.load_state_dict(
                saved_checkpoint_state[name]
            )

        return suffix
