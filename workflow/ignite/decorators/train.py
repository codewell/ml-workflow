from functools import wraps

from workflow.torch import model_device
from workflow.ignite.decorators import (
    to_device, step
)


def train(model, optimizer, n_batches_per_step=1):
    device = model_device(model)

    def decorator(process_batch):

        @wraps(process_batch)
        @to_device(device)
        @step(optimizer, n_batches_per_step=n_batches_per_step)
        def _process_batch(*args, **kwargs):
            model.train()
            return process_batch(*args, **kwargs)

        return _process_batch

    return decorator
