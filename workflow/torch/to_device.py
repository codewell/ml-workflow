import torch


def to_device(x, device):
    if type(x) == tuple:
        return tuple(to_device(value, device) for value in x)
    if type(x) == list:
        return [to_device(value, device) for value in x]
    elif type(x) == dict:
        return {key: to_device(value, device) for key, value in x.items()}
    elif type(x) == torch.Tensor:
        return x.to(device)
    else:
        raise Exception(f'Unknown type {type(x)}, cannot move to device')
