

def step(optimizer, batches_per_step=1):
    def decorator(backward_fn):
        def batch_fn(engine, batch):

            result = backward_fn(engine, batch)

            if engine.state.iteration % batches_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            return result
        return batch_fn
    return decorator
