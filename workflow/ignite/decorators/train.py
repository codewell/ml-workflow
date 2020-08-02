import torch
from functools import wraps

from workflow.functional import structure_map
from workflow.torch import module_device, module_train
from workflow.ignite.decorators import (
    to_device, step
)


def cpu_detach(x):
    if type(x) is torch.Tensor:
        return x.detach().cpu()
    else:
        return x


def train(model, optimizer, n_batches_per_step=1):
    device = module_device(model)

    def decorator(process_batch):

        @wraps(process_batch)
        @to_device(device)
        @step(optimizer, n_batches_per_step=n_batches_per_step)
        def _process_batch(*args, **kwargs):
            with module_train(model):
                return structure_map(
                    cpu_detach,
                    process_batch(*args, **kwargs),
                )

        return _process_batch

    return decorator
