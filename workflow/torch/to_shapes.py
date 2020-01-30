import torch


def to_shapes(*args):
    '''
    Helper function to convert nested dict / list of tensors to shapes
    for debugging
    '''
    if len(args) == 1:
        x = args[0]
    else:
        x = args

    if type(x) == tuple:
        return tuple(to_shapes(value) for value in x)
    if type(x) == list:
        return [to_shapes(value) for value in x]
    elif type(x) == dict:
        return {key: to_shapes(value) for key, value in x.items()}
    elif hasattr(x, 'shape'):
        return x.shape
    else:
        return x
