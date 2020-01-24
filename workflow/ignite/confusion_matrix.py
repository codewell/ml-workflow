from sklearn.metrics import confusion_matrix
import ignite.metrics
import numpy as np


class ConfusionMatrix(ignite.metrics.Metric):

    def __init__(self, num_classes, output_transform=lambda x: x):
        self.num_classes = num_classes
        super().__init__(output_transform=output_transform)

    def compute(self):
        return self.matrix

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, output):
        self.matrix += confusion_matrix(
            output['targets'].cpu(), output['predictions'].cpu(),
            labels=list(range(self.num_classes))
        )
