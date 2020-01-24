from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Events
from .attach_lr_warmup import attach_lr_warmup
from torch.optim.lr_scheduler import StepLR


def attach_exponential_decay_lr(trainer, optimizer, config):

    scheduler = LRScheduler(
        StepLR(optimizer, config['steps_per_epoch'], gamma=config['lr_decay'])
    )
    if config['warmup_lr']:
        attach_lr_warmup(trainer, config, scheduler)
    else:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
