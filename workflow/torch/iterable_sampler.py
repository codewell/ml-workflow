import torch


class IterableSampler(torch.utils.data.Sampler):
    def __init__(self, iterable):
        super().__init__(data_source=None)
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def __len__(self):
        raise Exception('IterableSampler has no length')
