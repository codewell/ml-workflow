import os
import torch
import numpy as np

from workflow.torch import model_device


def load_best_model(
    checkpoint_dir,
    checkpoint,
    device=None,
    prefix='model_',
    suffix=None,
):
    models = os.listdir(checkpoint_dir)
    if suffix is None:
        suffixes = [
            '_'.join(
                os.path.splitext(name)[0]
                .lstrip(prefix).split('_')[1:]
            )
            for name in models
        ]
        suffix = suffixes[np.argmax([float(s.split('_')[-1]) for s in suffixes])]

    saved_checkpoint_state = torch.load(
        f'{checkpoint_dir}/{prefix}checkpoint_{suffix}.pth',
        map_location=device,
    )

    for name, module_or_optimizer in checkpoint.items():
        module_or_optimizer.load_state_dict(
            saved_checkpoint_state[name]
        )

    return suffix
