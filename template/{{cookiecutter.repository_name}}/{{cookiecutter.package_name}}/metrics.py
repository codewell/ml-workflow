import ignite


def metrics():
    return dict(
        loss=ignite.metrics.Average(
            lambda output: output['loss']
        ),
        accuracy=ignite.metrics.Accuracy(lambda output: (
            output['predicted_class_name'],
            output['class_name'],
        )),
    )
