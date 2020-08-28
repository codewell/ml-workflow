from functools import wraps, partial


class RequiresGrad:
    '''Can be used as a decorator or context manager'''
    def __init__(self, module, grad=True):
        if type(grad) is not bool:
            raise TypeError('Input argument "grad" must be bool')

        self.module = module
        self.grad = grad

    def __enter__(self):
        self.named_requires_grads = {name: parameter.requires_grad for name, parameter in self.module.named_parameters()}

        for parameter in self.module.parameters():
            parameter.requires_grad = self.grad

        return self.module

    def __exit__(self, type, value, traceback):
        for name, parameter in self.module.named_parameters():
            parameter.requires_grad = self.named_requires_grads[name]

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with RequiresGrad(self.module, self.grad):
                return fn(*args, **kwargs)
        return wrapper


requires_grad = RequiresGrad
requires_nograd = partial(RequiresGrad, grad=False)


def test_requires_grad():
    import torch.nn as nn

    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(1, 1)

            for param in self.parameters():
                param.requires_grad = False

        def forward(self, x):
            return self.a(x)

    model = A()
    assert(all(map(lambda x: not x.requires_grad, model.parameters())))

    with requires_grad(model):
        assert(all(map(lambda x: x.requires_grad, model.parameters())))
    assert(all(map(lambda x: not x.requires_grad, model.parameters())))


def test_nested():
    import torch.nn as nn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(1, 1)

        def forward(self, x):
            return self.a(x)

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(1, 1)
            self.inner = Inner()

            for param in self.parameters():
                param.requires_grad = False

        def forward(self, x):
            return self.a(x)

    model = Outer()
    assert(all(map(lambda x: not x.requires_grad, model.parameters())))

    with requires_grad(model):
        with requires_nograd(model):
            assert(all(map(lambda x: not x.requires_grad, model.parameters())))
        assert(all(map(lambda x: x.requires_grad, model.parameters())))
    assert(all(map(lambda x: not x.requires_grad, model.parameters())))

    with requires_grad(model):
        with requires_nograd(model.inner):
            assert(all(map(lambda x: not x.requires_grad, model.inner.parameters())))
        assert(all(map(lambda x: x.requires_grad, model.parameters())))
    assert(all(map(lambda x: not x.requires_grad, model.parameters())))

    with requires_grad(model.inner):
        assert(all(map(lambda x: x.requires_grad, model.inner.parameters())))
        assert(any(map(lambda x: not x.requires_grad, model.parameters())))
        with requires_nograd(model):
            assert(all(map(lambda x: not x.requires_grad, model.parameters())))
        assert(all(map(lambda x: x.requires_grad, model.inner.parameters())))
    assert(all(map(lambda x: not x.requires_grad, model.parameters())))
