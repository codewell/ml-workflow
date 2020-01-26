

def model_device(model):
    return next(model.parameters()).device
