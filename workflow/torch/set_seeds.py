import random
import numpy as np
import torch


def set_seeds(seed):
    np.random.seed(seed=seed + 1)
    random.seed(seed + 2)
    torch.manual_seed(seed + 3)

    try:
        import imgaug
        imgaug.seed(seed + 4)
    except ImportError:
        pass
