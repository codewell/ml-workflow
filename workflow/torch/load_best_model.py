import os
import torch
import numpy as np


def load_best_model(
    checkpoint_dir,
    model,
    optimizer=None,
    prefix='model_',
    suffix=None
):
    models = os.listdir(checkpoint_dir)
    if suffix is None:
        suffixes = [
            '_'.join(os.path.splitext(name)[0].lstrip(prefix).split('_')[1:])
            for name in models
        ]
        suffix = suffixes[np.argmax([float(s.split('_')[-1]) for s in suffixes])]

    state = torch.load(
        f'{checkpoint_dir}/{prefix}checkpoint_{suffix}.pth',
        map_location=next(model.parameters()).device
    )

    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    return suffix
