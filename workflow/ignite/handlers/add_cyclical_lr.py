from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
from ignite.engine import Events
from .add_lr_warmup import add_lr_warmup


def add_cyclical_lr(trainer, optimizer, config):

    scheduler = LinearCyclicalScheduler(
        optimizer, 'lr',
        start_value=config['learning_rate'] / 100,
        end_value=config['learning_rate'],
        cycle_size=config['n_batches_per_epoch'],
        start_value_mult=config['lr_decay'],
        end_value_mult=config['lr_decay'],
    )

    if config['warmup_lr']:
        add_lr_warmup(trainer, config, scheduler)
    else:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
