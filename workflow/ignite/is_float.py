import numpy as np
import torch


def is_float(value):
    return type(value) in [
        float, np.float16, np.float32, np.float64, torch.float16,
        torch.float32, torch.float64
    ]
