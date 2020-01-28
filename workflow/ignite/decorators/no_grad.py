import torch


def no_grad(fn):
    def _fn(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return _fn
