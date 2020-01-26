import torch
import ignite
from workflow.torch import model_device, to_device


def get_evaluator(model):    
    device = model_device(model)

    def process_batch(engine, batch):
        model.eval()
        batch = to_device(batch, model)
        with torch.no_grad():
            output = model(batch['features'])

        return dict(
            output=output,
            targets=batch['targets'],
        )
    
    return ignite.engine.Engine(process_batch)
