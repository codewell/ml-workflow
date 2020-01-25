from itertools import chain

from workflow.functional.starcompose import starcompose


interleaved = starcompose(
    tuple,
    zip,
    chain.from_iterable,
)


def test_interleaved():
    from itertools import repeat

    test = interleaved([range(3)])
    if list(test) != [0, 1, 2]:
        raise Exception('Interleave with single iterator failed')

    test = interleaved([
        range(3),
        range(5)
    ])

    if list(test) != [0, 0, 1, 1, 2, 2]:
        raise Exception('Interleave with different lengths failed')

    if list(test) != []:
        raise Exception('Expected generator to be exhausted')
