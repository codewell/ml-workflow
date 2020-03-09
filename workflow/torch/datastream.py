from functools import partial
from itertools import repeat, chain, islice
import numpy as np
import pandas as pd
import torch
from workflow.functional import starcompose, star, repeat_map_chain
from workflow.torch.dataset import Dataset


class Sampler(torch.utils.data.Sampler):
    def __init__(self, fn, length):
        super().__init__(range(length))
        self.fn = fn
        self.length = length

    def __iter__(self):
        return islice(self.fn(), self.length)

    def __len__(self):
        return self.length


class Datastream:
    def __init__(self, dataset, sampler=None):
        super().__init__()
        self.dataset = dataset
        self._sampler = sampler

    @property
    def sampler(self):
        '''Weights not implemented'''
        if self._sampler is None:
            return torch.utils.data.RandomSampler(self.dataset)
        else:
            return self._sampler

    def data_loader(self, n_batches_per_epoch=None, **kwargs):
        if n_batches_per_epoch is None:
            sampler = self.sampler
        else:
            if hasattr(self.sampler, 'fn'):
                sampler_fn = self.sampler.fn
            else:
                sampler_fn = lambda: iter(self.sampler)

            sampler = Sampler(
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

        create_sampler = starcompose(
            partial(map, star(lambda sampler, n: (
                repeat_map_chain(iter, sampler),
                n,
            ))),
            partial(map, star(batch)),
            star(zip),
            partial(map, enumerate),
            chain.from_iterable,
            partial(map, star(lambda dataset_index, batch: zip(
                repeat(dataset_index), batch
            ))),
            chain.from_iterable,
            partial(map, star(map_index)),
            iter,
        )
        return create_sampler(samplers_and_ns)

    @staticmethod
    def merge(datastreams_and_ns):
        datastreams_and_ns = [
            x if type(x) is tuple else (x, 1)
            for x in datastreams_and_ns
        ]

        datasets = [datastream.dataset for datastream, n in datastreams_and_ns]
        concat_dataset = Dataset.concat(datasets)

        samplers_and_ns = [
            (datastream.sampler, n)
            for (datastream, n) in datastreams_and_ns
        ]
        sampler = Sampler(
            lambda: Datastream._merge_samplers(
                samplers_and_ns,
                Dataset.create_to_concat_mapping(datasets),
            ),
            length=max(map(len, datasets)),
        )

        return Datastream(
            concat_dataset,
            sampler,
        )

    def map(self, fn):
        return Datastream(
            self.dataset.map(fn),
            self._sampler,
        )


def test_datastream():

    datastream = Datastream.merge([
        Datastream(Dataset.from_indexable(list('abc'))),
        Datastream(Dataset.from_indexable(list('def'))),
    ])

    it = iter(datastream.sampler)

    for _ in range(2):
        index = next(it)
