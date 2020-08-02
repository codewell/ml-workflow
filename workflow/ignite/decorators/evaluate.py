import torch
from functools import wraps

from workflow.functional import structure_map
from workflow.torch import module_device, module_eval
from workflow.ignite.decorators.to_device import to_device


def cpu_detach(x):
    if type(x) is torch.Tensor:
        return x.detach().cpu()
    else:
        return x


def evaluate(model):
    device = module_device(model)

    def decorator(process_batch):
        
        @wraps(process_batch)
        @to_device(device)
        @torch.no_grad()
        def _process_batch(*args, **kwargs):
            with module_eval(model):
                return structure_map(
                    cpu_detach,
                    process_batch(*args, **kwargs),
                )
        return _process_batch
        
    return decorator
