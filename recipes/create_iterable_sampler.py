from functools import partial

from workflow.functional import starcompose, interleaved, repeat_map_chain
from workflow.torch import sample, IterableSampler


create_sampler = starcompose(
    lambda dataframe: [
        df.index.unique() for _, df in dataframe.groupby(['data_source', 'class_name'])
    ],
    partial(map, partial(repeat_map_chain, sample)),
    interleaved,
    IterableSampler,
)

sampler = create_sampler(dataframes['train'])
