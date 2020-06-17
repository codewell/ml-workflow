import numpy as np
import ignite


def metrics():
    return dict(
        loss=ignite.metrics.Average(
            lambda output: output['loss']
        ),
        accuracy=ignite.metrics.Average(lambda output: np.mean([
            prediction.class_name() == example.class_name
            for prediction, example in zip(
                output['predictions'], output['examples']
            )
        ])),
    )
