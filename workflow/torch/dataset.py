from functools import partial
from itertools import repeat, chain
import numpy as np
import pandas as pd
import torch
from workflow.functional import starcompose, star


class Dataset(torch.utils.data.Dataset):
    def __init__(self, iterable, function_list):
        super().__init__()
        self.iterable = iterable
        self.function_list = function_list
        self.composed_fn = starcompose(*function_list)

    @staticmethod
    def from_iterable(iterable):
        return Dataset(
            iterable,
            [lambda ds, index: ds[index]],
        )

    @staticmethod
    def from_dataset(dataset):
        return Dataset.from_iterable(dataset)

    @staticmethod
    def from_dataframe(dataframe):
        return Dataset(
            dataframe,
            [lambda df, index: df.iloc[index]],
        )

    @property
    def dataframe(self):
        return pd.DataFrame(self.iterable)

    def __getitem__(self, index):
        return self.composed_fn(self.iterable, index)

    def __len__(self):
        return len(self.iterable)

    def __str__(self):
        return str('\n'.join(
            [str(self[index]) for index in range(min(3, len(self)))]
            + ['...'] if len(self) > 3 else []
        ))

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return Dataset.concat([self, other])

    def map(self, fn):
        return Dataset(
            self.iterable,
            self.function_list + [fn],
        )

    def subset(self, indices):
        return Dataset(
            indices,
            (
                [lambda indices, outer_index: (
                    self.iterable, indices[outer_index]
                )]
                + self.function_list
            ),
        )

    def filter(self, filter_fn):
        '''
        Filter is not lazy. It must filter the items before hand to allow
        the sampler to pick unfiltered items.
        '''

        return self.subset(np.arange(len(self))[
            list(map(
                partial(
                    starcompose(*self.function_list, filter_fn),
                    self.iterable,
                ),
                range(len(self)),
            ))
        ])

    @staticmethod
    def create_concat_mapping(datasets):
        return starcompose(
            enumerate,
            partial(map, star(lambda dataset_index, iterable: zip(
                repeat(dataset_index),
                range(len(iterable)),
            ))),
            chain.from_iterable,
            list,
        )(datasets)

    @staticmethod
    def concat(datasets):
        return Dataset(
            Dataset.create_concat_mapping(datasets),
            [
                lambda concat_mapping, index: concat_mapping[index],
                lambda dataset_index, inner_index: (
                    datasets[dataset_index][inner_index]
                ),
            ]
        )


def test_dataset():
    dataset = Dataset.concat([
        Dataset.from_iterable(list(range(5))),
        Dataset.from_iterable(list(range(4))),
    ])
