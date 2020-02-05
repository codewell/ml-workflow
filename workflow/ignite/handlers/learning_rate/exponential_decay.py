from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR


class ExponentialDecay(LRScheduler):
    def __init__(self, optimizer, config):
        super().__init__(
            StepLR(
                optimizer,
                config['n_batches_per_epoch'],
                gamma=config['lr_decay']
            )
        )
