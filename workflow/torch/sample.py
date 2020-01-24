import torch


def sample(dataset, sampler=torch.utils.data.RandomSampler):
    return map(
        dataset.__getitem__,
        sampler(dataset)
    )
