import ignite
from workflow.ignite import create_standard_trainer_validator
from workflow.ignite.decorators import train, evaluate


@train(model, optimizer, n_batches_per_step)
def train_batch(engine, batch):
    batch['prediction'] = model(batch['image'])
    loss_ = F.binary_cross_entropy_with_logits(
        batch['prediction'], batch['label']
    )
    loss_.backward()
    batch['loss'] = loss_.item()
    return batch


@evaluate(model)
def evaluate_batch(engine, batch):
    batch['prediction'] = model(batch['image'])
    batch['loss'] = F.binary_cross_entropy_with_logits(
        batch['prediction'], batch['label']
    ).item()
    return batch


def loss_output_transform(output):
    return output['loss']


def metric_output_transform(output):
    return (
        output['prediction'],
        output['label'].argmax(-1),
    )


trainer, validator = create_standard_trainer_validator(
    model,
    optimizer,
    train_batch,
    evaluate_batch,
    validate_data_loader,
    model_score_function=lambda engine: -engine.state.metrics['loss'],
    trainer_metrics=dict(
        batch_loss=ignite.metrics.RunningAverage(
            output_transform=loss_output_transform, alpha=1e-5
        ),
        running_loss=ignite.metrics.RunningAverage(
            output_transform=loss_output_transform
        ),
    ),
    validator_metrics=dict(
        loss=ignite.metrics.Average(loss_output_transform),
        accuracy=ignite.metrics.Accuracy(metric_output_transform),
        precision=ignite.metrics.Precision(metric_output_transform),
        recall=ignite.metrics.Recall(metric_output_transform),
        confusion_matrix=ignite.metrics.ConfusionMatrix(
            num_classes=2,
            output_transform=metric_output_transform,
        ),
    ),
    config=config,
)
