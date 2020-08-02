from contextlib import contextmanager
from functools import partial


@contextmanager
def module_train(module, training=True):
    '''
    Usage:
    with module_train(model):
        prediction = model(features)
    '''
    was_training = module.training

    module.train(training)
    yield module

    module.train(was_training)


module_eval = partial(module_train, training=False)
