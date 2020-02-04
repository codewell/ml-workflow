from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Events
from .add_lr_warmup import add_lr_warmup
from torch.optim.lr_scheduler import StepLR


def add_exponential_decay_lr(trainer, optimizer, config):

    scheduler = LRScheduler(
        StepLR(
            optimizer,
            config['n_batches_per_epoch'],
            gamma=config['lr_decay']
        )
    )
    if config['warmup_lr']:
        add_lr_warmup(trainer, config, scheduler)
    else:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
