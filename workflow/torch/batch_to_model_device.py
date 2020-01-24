from workflow.torch.to_device import to_device


def batch_to_model_device(batch, model):
    device = next(model.parameters()).device
    if type(batch) in (list, dict):
        return to_device(batch, device)
    else:
        raise ValueError('Expected batch to be type list or dict')
