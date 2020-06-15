import torch


def binary_cross_entropy(predicted, target):
    '''
    Binary cross entropy implementation that allows targets with values between
    0 and 1.

    https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/ops/nn_impl.py#L113
    '''
    return (
        torch.relu(predicted) - predicted * target
        + torch.where(predicted >= 0, -predicted, predicted).exp().log1p()
    )
