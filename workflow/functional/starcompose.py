

def starcompose(*transforms):
    '''
    left compose functions together and expand tuples to args

    Use starcompose.debug for verbose output when debugging
    '''
    def _compose(*x):
        for t in transforms:
            if type(x) is tuple:
                x = t(*x)
            else:
                x = t(x)
        return x
    return _compose


def starcompose_debug(*transforms):
    '''
    verbose starcompose for debugging
    '''
    print('starcompose debug')
    def _compose(*x):
        for index, t in enumerate(transforms):
            print(f'{index}:, fn={t}, x={x}')
            if type(x) is tuple:
                x = t(*x)
            else:
                x = t(x)
        return x
    return _compose

starcompose.debug = starcompose_debug


def test_starcompose():
    from functools import partial

    test = starcompose(
        zip,
        partial(map, sum),
        list,
    )

    test(range(10), range(10))
