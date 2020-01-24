from itertools import chain

from workflow.functional.starcompose import starcompose


interleaved = starcompose(
    tuple,
    zip,
    chain.from_iterable,
)
