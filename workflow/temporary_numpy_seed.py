import numpy as np


def temporary_numpy_seed(fn, seed=None):
    random_state = np.random.get_state()
    np.random.seed(seed)
    output = fn()
    np.random.set_state(random_state)
    return output
