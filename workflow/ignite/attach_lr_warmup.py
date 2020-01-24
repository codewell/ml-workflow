import ignite.contrib.handlers
from ignite.contrib.handlers.param_scheduler import create_lr_scheduler_with_warmup
from ignite.engine import Events


def attach_lr_warmup(trainer, config, lr_scheduler):

    warmup_duration = (
        config['warmup_duration'] if config['warmup_duration'] > 0
        else config['steps_per_epoch'] * -config['warmup_duration']
    )

    warmup_end_value = (
        config['warmup_end_value'] if config['warmup_end_value'] != -1
        else config['learning_rate']
    )

    scheduler_with_warmup = create_lr_scheduler_with_warmup(
        lr_scheduler,
        config['warmup_start_value'],
        warmup_end_value,
        warmup_duration,
    )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_with_warmup)
