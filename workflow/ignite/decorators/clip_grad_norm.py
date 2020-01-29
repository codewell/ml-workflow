from functools import wraps

import torch


def clip_grad_norm(model, max_norm, norm_type=2):
    def decorator(fn):
        @wraps(fn)
        def _fn(*args, **kwargs):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm, norm_type=norm_type
            )
            return fn(*args, **kwargs)
        return _fn
    return decorator
