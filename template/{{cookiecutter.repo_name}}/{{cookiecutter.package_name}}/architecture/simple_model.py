from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from workflow.torch import ModuleCompose


def SimpleModel():
    return ModuleCompose(
        nn.Conv2d(1, 32, 3, 1),
        F.relu,
        nn.Conv2d(32, 64, 3, 1),
        F.relu,
        partial(F.max_pool2d, kernel_size=2),
        nn.Dropout2d(0.25),
        partial(torch.flatten, start_dim=1),
        nn.Linear(9216, 128),
        F.relu,
        nn.Dropout2d(0.5),
        nn.Linear(128, 10),
        partial(F.log_softmax, dim=1),
    )
