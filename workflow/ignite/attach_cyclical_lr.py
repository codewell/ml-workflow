from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
from ignite.engine import Events
from .attach_lr_warmup import attach_lr_warmup


def attach_cyclical_lr(trainer, optimizer, config):

    scheduler = LinearCyclicalScheduler(
        optimizer, 'lr',
        start_value=config['learning_rate'] / 100,
        end_value=config['learning_rate'],
        cycle_size=config['steps_per_epoch'],
        start_value_mult=config['lr_decay'],
        end_value_mult=config['lr_decay'],
    )

    if config['warmup_lr']:
        attach_lr_warmup(trainer, config, scheduler)
    else:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
