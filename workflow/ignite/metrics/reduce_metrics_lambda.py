import numpy as np
from ignite.metrics import MetricsLambda


class ReduceMetricsLambda(MetricsLambda):
    def __init__(self, reduce_fn, f, *args, **kwargs):
        super().__init__(f, *args, **kwargs)
        self.reduce_fn = reduce_fn
        self.reduced_value = None

    def compute(self):
        value = super().compute()
        if self.reduced_value is None:
            self.reduced_value = value
        else:
            self.reduced_value = self.reduce_fn(self.reduced_value, value)

        return self.reduced_value
