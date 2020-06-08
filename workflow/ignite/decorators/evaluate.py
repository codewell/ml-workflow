import torch
from functools import wraps

from workflow.functional import structure_map
from workflow.torch import model_device
from workflow.ignite.decorators.to_device import to_device


def detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    else:
        return x


def evaluate(model):
    device = model_device(model)

    def decorator(process_batch):
        
        @wraps(process_batch)
        @to_device(device)
        @torch.no_grad()
        def _process_batch(*args, **kwargs):
            model.eval()
            return structure_map(
                detach,
                process_batch(*args, **kwargs),
            )
        return _process_batch
        
    return decorator
