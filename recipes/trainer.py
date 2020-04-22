import ignite
from workflow.ignite import trainer as workflow_trainer
from workflow.ignite.decorators import train, evaluate


@workflow.ignite.decorators.train(model, optimizer, n_batches_per_step)
def train_batch(engine, batch):
    batch['prediction'] = model(batch['image'])
    loss_ = F.binary_cross_entropy_with_logits(
        batch['prediction'], batch['label']
    )
    loss_.backward()
    batch['loss'] = loss_.item()
    return batch


@workflow.ignite.decorators.evaluate(model)
def evaluate_batch(engine, batch):
    batch['prediction'] = model(batch['image'])
    batch['loss'] = F.binary_cross_entropy_with_logits(
        batch['prediction'], batch['label']
    ).item()
    return batch


def loss_output_transform(output):
    return output['loss']


trainer, validator = workflow_trainer(
    train_batch,
    evaluate_batch,
    evaluate_data_loaders,
    metrics=dict(
        train=dict(
            batch_loss=workflow_trainer.Progress(ignite.metrics.RunningAverage(
                output_transform=loss_output_transform, alpha=1e-5, epoch_bound=False
            )),
            loss=ignite.metrics.Average(
                output_transform=loss_output_transform
            ),
        )
    ),
    optimizers=dict(optimizer=optimizer),
)
