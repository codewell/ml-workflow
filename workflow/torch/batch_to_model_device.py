from workflow.torch.to_device import to_device


def batch_to_model_device(batch, model):
    return to_device(batch, next(model.parameters()).device)
