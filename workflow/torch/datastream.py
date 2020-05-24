from functools import partial
from itertools import repeat, chain, islice
import numpy as np
import pandas as pd
import torch
from workflow.functional import starcompose, star, repeat_map_chain
from workflow.torch.dataset import Dataset


class DatastreamSampler(torch.utils.data.Sampler):
    def __init__(self, fn, length):
        super().__init__(range(length))
        self.fn = fn
        self.length = length
        self.queue = self.fn()

    def __iter__(self):
        for _ in range(self.length):
            try:
                yield next(self.queue)
            except StopIteration:
                self.queue = self.fn()
                yield next(self.queue)

    def __len__(self):
        return self.length


class Datastream:
    def __init__(self, dataset, sampler=None):
        super().__init__()
        self.dataset = dataset

        if sampler is None:
            sampler = torch.utils.data.RandomSampler(self.dataset)
        self.sampler = sampler

    def data_loader(self, n_batches_per_epoch=None, **kwargs):
        if n_batches_per_epoch is None:
            sampler = self.sampler
        else:
            if hasattr(self.sampler, 'fn'):
                sampler_fn = self.sampler.fn
            else:
                sampler_fn = lambda: iter(self.sampler)

            sampler = DatastreamSampler(
                sampler_fn, n_batches_per_epoch * kwargs['batch_size']
            )

        return torch.utils.data.DataLoader(
            self.dataset, sampler=sampler, **kwargs
        )

    @staticmethod
    def _merge_samplers(samplers_and_ns, map_index):
        def batch(iterable, n):
            while True:
                yield [next(iterable) for _ in range(n)]


        index_batch = zip(*[
            batch(map(
                partial(map_index, dataset_index),
                repeat_map_chain(iter, sampler),
            ), n)
            for dataset_index, (sampler, n) in enumerate(samplers_and_ns)
        ])

        return chain.from_iterable(chain.from_iterable(index_batch))

    @staticmethod
    def merge(datastreams_and_ns):
        datastreams_and_ns = [
            x if type(x) is tuple else (x, 1)
            for x in datastreams_and_ns
        ]

        datasets = [datastream.dataset for datastream, n in datastreams_and_ns]
        samplers_and_ns = [
            (datastream.sampler, n)
            for (datastream, n) in datastreams_and_ns
        ]
        sampler = DatastreamSampler(
            lambda: Datastream._merge_samplers(
                samplers_and_ns,
                Dataset.create_to_concat_mapping(datasets),
            ),
            length=max(map(len, datasets)) * len(datasets),
        )

        return Datastream(
            Dataset.concat(datasets),
            sampler,
        )

    @staticmethod
    def _zip_samplers(samplers, map_index):
        create_sampler = starcompose(
            partial(map, partial(repeat_map_chain, iter)),
            tuple,
            zip,
            partial(map, map_index),
        )
        return create_sampler(samplers)

    @staticmethod
    def zip(datastreams):
        datasets = [datastream.dataset for datastream in datastreams]
        samplers = [datastream.sampler for datastream in datastreams]
        sampler = DatastreamSampler(
            lambda: Datastream._zip_samplers(
                samplers,
                Dataset.create_to_combine_mapping(datasets),
            ),
            length=max(map(len, datasets)),
        )
        return Datastream(
            Dataset.combine(datasets),
            sampler,
        )

    def map(self, fn):
        return Datastream(
            self.dataset.map(fn),
            self.sampler,
        )


def test_datastream_merge():

    datastream = Datastream.merge([
        Datastream(Dataset.from_subscriptable(list('abc'))),
        Datastream(Dataset.from_subscriptable(list('def'))),
    ])

    it = iter(datastream.sampler)
    for _ in range(2):
        index = next(it)

    batch = next(iter(datastream.data_loader(batch_size=8)))


def test_datastream_zip():

    datasets = [
        Dataset.from_subscriptable([1, 2]),
        Dataset.from_subscriptable([3, 4, 5]),
        Dataset.from_subscriptable([6, 7]),
    ]

    datastreams = [
        Datastream(ds, sampler=torch.utils.data.SequentialSampler(ds))
        for ds in datasets
    ]
    zipped_datastream = Datastream.zip(datastreams)

    batch = next(iter(zipped_datastream.data_loader(batch_size=3)))
    assert len(batch) == 3 and len(batch[0]) == 3
    assert batch[0][0] == 1 and batch[0][1] == 2 and batch[0][2] == 1
    assert batch[1][0] == 3 and batch[1][1] == 4 and batch[1][2] == 5
    assert batch[2][0] == 6 and batch[2][1] == 7 and batch[2][2] == 6
