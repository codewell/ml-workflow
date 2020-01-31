import torch
from itertools import islice


class IterableSampler(torch.utils.data.Sampler):
    def __init__(self, iterable, length=None):
        super().__init__(data_source=None)
        self.iterable = iterable
        self.iterator = iter(iterable)
        self.length = length

    def __iter__(self):
        return islice(self.iterator, self.length)

    def __len__(self):
        if self.length is None:
            raise Exception('Length is none')
        else:
            return self.length
