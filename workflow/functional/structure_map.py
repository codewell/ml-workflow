

def structure_map(fn, x):
    if type(x) is tuple:
        return tuple(structure_map(fn, value) for value in x)
    elif type(x) is list:
        return [structure_map(fn, value) for value in x]
    elif type(x) is dict:
        return {key: structure_map(fn, value) for key, value in x.items()}
    else:
        return fn(x)
