import numpy as np

from ignite.metrics import (
    ConfusionMatrix,
    MetricsLambda,
)


def _matthews_correlation_coefficient(confusion_matrix):
    [[tp, fp], [fn, tn]] = confusion_matrix

    if all([
        0 < tp + fp,
        0 < tp + fn,
        0 < tn + fp,
        0 < tn + fn,
    ]):
        return (
            (tp * tn - fp * fn)
                / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )
    else:
        return 0.

def MatthewsCorrelationCoefficient(output_transform):
    return MetricsLambda(
        _matthews_correlation_coefficient,
        ConfusionMatrix(num_classes=2, output_transform=output_transform)
    )
