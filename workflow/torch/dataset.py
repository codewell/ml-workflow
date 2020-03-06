from functools import partial
from itertools import repeat, chain
import numpy as np
import pandas as pd
import torch
from workflow.functional import starcompose, star


class Dataset(torch.utils.data.Dataset):
    def __init__(self, source, length, function_list):
        super().__init__()
        self.source = source
        self.length = length
        self.function_list = function_list
        self.composed_fn = starcompose(*function_list)

    @staticmethod
    def from_indexable(indexable):
        return Dataset(
            indexable,
            len(indexable),
            [lambda ds, index: ds[index]],
        )

    @staticmethod
    def from_dataframe(dataframe):
        return Dataset(
            dataframe,
            len(dataframe),
            [lambda df, index: df.iloc[index]],
        )

    def __getitem__(self, index):
        return self.composed_fn(self.source, index)

    def __len__(self):
        return self.length

    def __str__(self):
        return str('\n'.join(
            [str(self[index]) for index in range(min(3, len(self)))]
            + ['...'] if len(self) > 3 else []
        ))

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return Dataset.concat([self, other])

    def map(self, function):
        return Dataset(
            self.source,
            self.length,
            self.function_list + [function],
        )

    def subset(self, indices):
        return Dataset(
            indices,
            len(indices),
            [lambda indices, outer_index: (
                self.source, indices[outer_index]
            )] + self.function_list,
        )

    @staticmethod
    def create_from_concat_mapping(datasets):
        cumulative_lengths = np.cumsum(list(map(len, datasets)))

        def from_concat(index):
            dataset_index = np.sum(index >= cumulative_lengths)
            if dataset_index == 0:
                inner_index = index
            else:
                inner_index = index - cumulative_lengths[dataset_index - 1]

            return dataset_index, inner_index
        return from_concat

    @staticmethod
    def create_to_concat_mapping(datasets):
        cumulative_lengths = np.cumsum(list(map(len, datasets)))

        def to_concat(dataset_index, inner_index):
            if dataset_index == 0:
                index = inner_index
            else:
                index = inner_index + cumulative_lengths[dataset_index - 1]

            return index
        return to_concat

    @staticmethod
    def concat(datasets):
        from_concat_mapping = Dataset.create_from_concat_mapping(datasets)

        return Dataset(
            datasets,
            sum(map(len, datasets)),
            [
                lambda datasets, index: (
                    datasets,
                    *from_concat_mapping(index),
                ),
                lambda datasets, dataset_index, inner_index: (
                    datasets[dataset_index][inner_index]
                ),
            ],
        )

    @staticmethod
    def zip(datasets):
        return Dataset(
            datasets,
            min(map(len, datasets)),
            [lambda datasets, index: tuple(
                dataset[index] for dataset in datasets
            )],
        )


def test_concat_dataset():
    dataset = Dataset.concat([
        Dataset.from_indexable(list(range(5))),
        Dataset.from_indexable(list(range(4))),
    ])

    if dataset[6] != 1:
        raise AssertionError('Unexpected result from Dataset.concat')


def test_zip_dataset():
    dataset = Dataset.zip([
        Dataset.from_indexable(list(range(5))),
        Dataset.from_indexable(list(range(4))),
    ])

    if dataset[3] != (3, 3):
        raise AssertionError('Unexpected result from Dataset.zip')
