from .tqdm_print import tqdm_print
from .is_float import is_float


def write_engine_metrics(engine, name='Training'):

    max_len = max(len(metric) for metric in engine.state.metrics.keys())
    for metric, value in engine.state.metrics.items():
        padding = ' ' * (max_len - len(metric))
        if hasattr(value, '__len__'):
            tqdm_print(f'{name} {metric}:')
            tqdm_print(str(value))
        elif is_float(value):
            tqdm_print(f'{name} {metric}:{padding} {value:.4f}')
        else:
            tqdm_print(f'{name} {metric}:{padding} {value}')
