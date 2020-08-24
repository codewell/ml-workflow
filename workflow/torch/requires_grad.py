from contextlib import contextmanager


@contextmanager
def requires_grad(module, grad=True):
    '''
    Usage:
    with requires_grad(model):
        prediction = model(features)
    '''
    requires_grads = [
        p.requires_grad for p in module.parameters()
    ]
    for p in module.parameters():
        p.requires_grad = grad

    yield module

    for p, requires_grad in zip(module.parameters(), requires_grads):
        p.requires_grad = requires_grad


def requires_nograd(module):
    '''
    Usage:
    with requires_nograd(model):
        prediction = model(features)
    '''
    return requires_grad(module, grad=False)
