from ignite.engine import Events
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR


class LearningRateScheduler(LRScheduler):
    def __init__(self, optimizer, multiplicative_fn):
        '''
        Usage:

        LearningRateScheduler(
            optimizer,
            starcompose(
                warmup(n_steps=100),
                cyclical(length=50, relative_minimum=0.1),
                decay(0.97),
            )
        ).attach(trainer)
        '''
        self.multiplicative_fn = multiplicative_fn
        super().__init__(
            LambdaLR(optimizer, self.learning_rate)
        )

    def learning_rate(self, step):
        _, learning_rate = self.multiplicative_fn(step, 1)
        return learning_rate

    def attach(self, trainer):
        trainer.add_event_handler(Events.ITERATION_STARTED, self.__call__)
