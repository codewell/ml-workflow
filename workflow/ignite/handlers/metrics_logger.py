from ignite.engine import Events

from workflow.ignite.tqdm_print import tqdm_print
from workflow.ignite.is_float import is_float


class MetricsLogger:
    def __init__(self, name):
        self.name = name

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._print)

    def _print(self, engine):
        if len(engine.state.metrics) >= 1:
            tqdm_print(f'{self.name}:')

            target_len = 18
            for metric, value in engine.state.metrics.items():
                padding = ' ' * (target_len - len(metric))
                if hasattr(value, '__len__'):
                    tqdm_print(f'  {metric}:')
                    tqdm_print(str(value))
                elif is_float(value):
                    tqdm_print(f'  {metric}:{padding} {value:.4f}')
                else:
                    tqdm_print(f'  {metric}:{padding} {value}')
