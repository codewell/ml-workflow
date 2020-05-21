from functools import wraps


def update_cpu_model(cpu_model, model):
    def decorator(process_batch):

        @wraps(process_batch)
        def wrapper(*args, **kwargs):
            output = process_batch(*args, **kwargs)
            cpu_model.load_state_dict({key: value.cpu() for key, value in model.state_dict().items()})
            return output
        return wrapper
    return decorator
