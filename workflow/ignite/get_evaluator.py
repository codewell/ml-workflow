import torch
import ignite
import workflow


def get_evaluator(model):
    
    def evaluate_step(engine, batch):

        model.eval()
        batch = workflow.torch.batch_to_model_device(batch, model)
        with torch.no_grad():
            output = model(batch['features'])

        return dict(
            output=output,
            targets=batch['targets'],
        )
    
    return ignite.engine.Engine(evaluate_step)
