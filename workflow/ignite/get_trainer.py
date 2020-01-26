import torch
import ignite
from .handlers.attach_train_progress_bar import attach_train_progress_bar
from workflow.torch import model_device, to_device


def get_trainer(model, criterion, optimizer, config, track_loss=True):
    device = model_device(model)

    def process_batch(engine, batch):
        model.train()
        n_batches_per_step = config.get('n_batches_per_step', 1)

        if engine.state.iteration % n_batches_per_step == 1:
            optimizer.zero_grad()

        batch = to_device(batch, device)
        output = model(batch['features'])
        loss = criterion(output, batch['targets']) / n_batches_per_step
        loss.backward()

        if engine.state.iteration % n_batches_per_step == 0:
            if config.get('clip_norm', False):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['clip_norm']
                )
            optimizer.step()

        return dict(loss=loss.item() * n_batches_per_step)

    trainer = ignite.engine.Engine(process_batch)
    
    if track_loss:
        ignite.metrics.RunningAverage(
            output_transform=lambda x: x['loss'], alpha=0.98
        ).attach(trainer, 'running avg loss')

    attach_train_progress_bar(trainer, config)

    trainer.add_event_handler(
        ignite.engine.Events.ITERATION_COMPLETED,
        ignite.handlers.TerminateOnNan(),
    )

    return trainer
