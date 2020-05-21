import random
import numpy as np
import imgaug
import torch


def set_seeds(seed):
    np.random.seed(seed=seed + 1)
    random.seed(seed + 2)
    torch.manual_seed(seed + 3)
    imgaug.seed(seed + 4)
