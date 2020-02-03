from .tqdm_print import tqdm_print
from .is_float import is_float


def write_engine_metrics(engine, name):

    if len(engine.state.metrics) >= 1:
        tqdm_print(f'{name}:')

        max_len = max(map(len, engine.state.metrics.keys()))
        for metric, value in engine.state.metrics.items():
            padding = ' ' * (max_len - len(metric))
            if hasattr(value, '__len__'):
                tqdm_print(f'  {metric}:')
                tqdm_print(str(value))
            elif is_float(value):
                tqdm_print(f'  {metric}:{padding} {value:.4f}')
            else:
                tqdm_print(f'  {metric}:{padding} {value}')
