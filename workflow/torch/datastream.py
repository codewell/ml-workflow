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
    def _interleave_samplers(samplers, map_index):
        create_sampler = starcompose(
            partial(map, partial(repeat_map_chain, iter)),
            tuple,
            zip,
            partial(map, enumerate),
            chain.from_iterable,
            partial(map, star(map_index)),
            iter,
        )
        return create_sampler(samplers)

    @staticmethod
    def interleave(datastreams):
        datasets = [datastream.dataset for datastream in datastreams]
        concat_dataset = Dataset.concat(datasets)

        samplers = [datastream.sampler for datastream in datastreams]
        sampler = Sampler(
            lambda: Datastream._interleave_samplers(
                samplers,
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

    datastream = Datastream.interleave([
        Datastream(Dataset.from_indexable(list('abc'))),
        Datastream(Dataset.from_indexable(list('def'))),
    ])

    it = iter(datastream.sampler)

    for _ in range(2):
        index = next(it)
