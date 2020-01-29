from functools import wraps


def step(optimizer, batches_per_step=1):
    def decorator(backward_fn):
        @wraps(backward_fn)
        def batch_fn(engine, batch):

            result = backward_fn(engine, batch)

            if engine.state.iteration % batches_per_step == 0:
                for param_group in optimizer.param_groups:
                    for parameters in param_group['params']:
                        if parameters.requires_grad:
                            parameters.grad.div_(batches_per_step)

                optimizer.step()
                optimizer.zero_grad()

            return result
        return batch_fn
    return decorator
