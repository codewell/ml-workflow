from ignite.contrib.handlers.param_scheduler import create_lr_scheduler_with_warmup


def add_warmup(lr_scheduler, config):
    return create_lr_scheduler_with_warmup(
        lr_scheduler,
        warmup_start_value=config['warmup_start_value'],
        warmup_end_value=config['learning_rate'],
        warmup_duration=int(config['n_batches_per_epoch'] * config['warmup_duration']),
    )
