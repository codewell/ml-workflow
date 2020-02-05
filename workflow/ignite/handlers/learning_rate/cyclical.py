from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler


class Cyclical(LinearCyclicalScheduler):
    def __init__(self, optimizer, config):
        super().__init__(
            optimizer,
            param_name='lr',
            start_value=config['learning_rate'] / 100,
            end_value=config['learning_rate'],
            cycle_size=config['n_batches_per_epoch'],
            start_value_mult=config['lr_decay'],
            end_value_mult=config['lr_decay'],
        )
