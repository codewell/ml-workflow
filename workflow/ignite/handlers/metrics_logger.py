from tqdm import tqdm
from ignite.engine import Events

from workflow.ignite.is_float import is_float


class MetricsLogger:
    def __init__(self, name):
        self.name = name

    def attach(self, engine, metric_names):
        engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda engine: self._print(engine, metric_names)
        )

    def _print(self, engine, metric_names):
        if len(engine.state.metrics) >= 1:
            tqdm.write(f'{self.name}:')
            
            target_len = max(map(len, metric_names))
            for metric_name in metric_names:
                value = engine.state.metrics.get(metric_name, None)

                if value is not None:
                    
                    padding = ' ' * (target_len - len(metric_name))
                    if hasattr(value, '__len__'):
                        tqdm.write(f'  {metric_name}:')
                        tqdm.write(str(value))
                    elif is_float(value):
                        tqdm.write(f'  {metric_name}:{padding} {value:.4f}')
                    else:
                        tqdm.write(f'  {metric_name}:{padding} {value}')
