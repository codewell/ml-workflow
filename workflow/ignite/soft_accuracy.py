import ignite.metrics


class SoftAccuracy(ignite.metrics.Metric):

    def __init__(self, output_transform=lambda x: x):
        self._num_correct = None
        self._num_examples = None
        super().__init__(output_transform=output_transform)

    def compute(self):
        return self._num_correct / self._num_examples

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        probabilities = output['probabilities'].to(output['targets'].device)
        correct = 1 - abs(output['targets'] - probabilities)
        self._num_correct += correct.sum().item()
        self._num_examples += correct.shape[0]
