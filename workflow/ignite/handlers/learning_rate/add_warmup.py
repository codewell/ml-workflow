from ignite.contrib.handlers.param_scheduler import create_lr_scheduler_with_warmup


def add_warmup(lr_scheduler, config):

    warmup_duration = (
        config['warmup_duration'] if config['warmup_duration'] > 0
        else config['steps_per_epoch'] * -config['warmup_duration']
    )

    warmup_end_value = (
        config['warmup_end_value'] if config['warmup_end_value'] != -1
        else config['learning_rate']
    )

    return create_lr_scheduler_with_warmup(
        lr_scheduler,
        config['warmup_start_value'],
        warmup_end_value,
        warmup_duration,
    )
