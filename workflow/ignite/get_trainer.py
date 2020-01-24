import torch
import ignite
import workflow


def get_trainer(model, criterion, optimizer, config, track_loss=True):

    def train_step(engine, batch):

        model.train()
        accumulation_steps = (
            1 if 'accumulation_steps' not in config
            else config['accumulation_steps']
        )

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.zero_grad()

        batch = workflow.torch.batch_to_model_device(batch, model)
        if 'mixup_alpha' in config and config['mixup_alpha'] > 0:
            batch = workflow.torch.mixup_batch(batch, alpha=config['mixup_alpha'])
        output = model(batch['features'])
        loss = criterion(output, batch['targets']) / accumulation_steps
        loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            if 'clip_norm' in config and config['clip_norm']:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['clip_norm']
                )
            optimizer.step()

        return dict(loss=loss.item() * accumulation_steps)

    trainer = ignite.engine.Engine(train_step)
    if track_loss:
        ignite.metrics.RunningAverage(
            output_transform=lambda x: x['loss'], alpha=0.98
        ).attach(trainer, 'running avg loss')

    return trainer
