from ignite.engine import Events
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR


class LearningRateScheduler(LRScheduler):
    def __init__(self, optimizer, base_learning_rate, relative_fn):
        '''
        Usage:

        LearningRateScheduler(
            optimizer,
            config['learning_rate'],
            starcompose(
                warmup(n_steps=100),
                cyclical(length=50, relative_minimum=0.1),
            )
        ).attach(trainer)
        '''
        self.base_learning_rate = base_learning_rate
        self.relative_fn = relative_fn
        super().__init__(
            LambdaLR(optimizer, self.learning_rate)
        )

    def learning_rate(self, step):
        _, learning_rate = self.relative_fn(step, self.base_learning_rate)
        return learning_rate

    def attach(self, trainer):
        trainer.add_event_handler(Events.ITERATION_STARTED, self.__call__)
