import numpy as np
import torch


def mixup_batch(batch, alpha=0.4):
    weight = np.random.beta(alpha, alpha)
    random_indices = torch.randperm(len(batch['features']))
    return dict(
        features=(
            batch['features'] * weight
            + (1 - weight) * batch['features'][random_indices]
        ),
        targets=(
            batch['targets'] * weight
            + (1 - weight) * batch['targets'][random_indices]
        )
    )
