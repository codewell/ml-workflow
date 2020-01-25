from torch.utils.tensorboard import SummaryWriter
import numpy as np
from ignite.engine import Events
from .loss_score_function import loss_score_function


def attach_best_results_logger(trainer, evaluator, score_function=None, prefix=''):

    writer = SummaryWriter()
    best_results = dict(score=-np.inf)
    if score_function is None:
        score_function = loss_score_function

    def update(engine):
        score = score_function(engine)
        if score > best_results['score']:
            for metric, value in engine.state.metrics.items():
                if hasattr(value, '__len__'):
                    continue
                metric = '_'.join(metric.split())
                writer.add_scalar(
                    f'{prefix}{metric}', value, trainer.state.epoch
                )
            best_results['score'] = score

    def close(engine):
        writer.close()

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, update)
    trainer.add_event_handler(Events.COMPLETED, close)
