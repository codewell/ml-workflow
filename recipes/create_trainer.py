from workflow.ignite.decorators import train


@train(model, optimizer, n_n_batches_per_step=2)
def train_batch(engine, batch):
    batch['predictions'] = model(batch['features'])
    loss_ = loss(batch)
    loss_.backward()
    batch['loss'] = loss_.item()
    return batch

trainer = ignite.engine.Engine(train_batch)

# add handlers and metrics
