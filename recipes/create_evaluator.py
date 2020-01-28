from workflow.ignite.decorators import evaluate

@evaluate(model)
def evaluate_batch(engine, batch):
    batch['predictions'] = model(batch['features'])
    return batch

evaluator = ignite.engine.Engine(evaluate_batch)

# add handlers and metrics
