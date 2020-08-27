from functools import wraps, partial


class RequiresGrad:
    '''Can be used as a decorator or context manager'''
    def __init__(self, module, grad=True):
        if type(grad) is not bool:
            raise TypeError('Input argument "grad" must be bool')

        self.module = module
        self.grad = grad
        self.requires_grads = [
            p.requires_grad for p in module.parameters()
        ]

    def __enter__(self):
        for p in self.module.parameters():
            p.requires_grad = self.grad
        return self.module

    def __exit__(self, type, value, traceback):
        for p, requires_grad in zip(
            self.module.parameters(), self.requires_grads
        ):
            p.requires_grad = requires_grad

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with RequiresGrad(self.module):
                return fn(*args, **kwargs)
        return wrapper

requires_grad = RequiresGrad
requires_nograd = partial(RequiresGrad, grad=False)
