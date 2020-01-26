import torch
import ignite
import workflow


def get_trainer(model, criterion, optimizer, config, track_loss=True):

    def process_batch(engine, batch):

        model.train()
        n_batches_per_step = config.get('n_batches_per_step', 1)

        if engine.state.iteration % n_batches_per_step == 0:
            optimizer.zero_grad()

        batch = workflow.torch.batch_to_model_device(batch, model)
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

    return trainer
