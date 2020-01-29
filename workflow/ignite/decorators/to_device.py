from functools import partial, wraps

from workflow.functional import starcompose
from workflow.torch import to_device as torch_to_device


def to_device(device):
    def decorator(fn):
        return starcompose(
            lambda *args: list(args),
            partial(torch_to_device, device=device),
            tuple,
            fn,
        )
    return decorator
