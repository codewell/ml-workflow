from functools import wraps, partial


class ModuleTrain:
    '''Can be used as a decorator or context manager'''
    def __init__(self, module, training=True):
        self.module = module
        self.training = training
        self.was_training = module.training

    def __enter__(self):
        self.module.train(self.training)
        return self.module

    def __exit__(self, type, value, traceback):
        self.module.train(self.was_training)

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with ModuleTrain(self.module, self.training):
                return fn(*args, **kwargs)
        return wrapper


module_train = ModuleTrain
module_eval = partial(ModuleTrain, training=False)
