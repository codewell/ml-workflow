import numpy as np
import ignite


def train_metrics():
    return dict(
        loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: output['loss'],
            epoch_bound=False,
        ),
    )


def progress_metrics():
    return  dict(
        batch_loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: output['loss'],
            epoch_bound=False,
            alpha=1e-7,
        ),
    )


def evaluate_metrics():
    return dict(
        loss=ignite.metrics.Average(
            lambda output: output['loss']
        ),
        accuracy=ignite.metrics.Average(lambda output: np.mean([
            prediction.class_name == example.class_name
            for prediction, example in zip(
                output['predictions'], output['examples']
            )
        ])),
    )
