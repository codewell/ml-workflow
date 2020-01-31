import math
from tqdm import tqdm
from ignite.engine import Events

from workflow.ignite.constants import TQDM_OUTFILE


def add_validation_progress_bar(evaluator, n_batches, config):

    validation_progress_bar = None

    epoch = 1

    def create_progress_bar():
        return tqdm(
            total=n_batches,
            file=TQDM_OUTFILE,
            desc=f'''Epoch [{epoch} / {config['max_epochs']}] Validate''',
            leave=False
        )

    validation_progress_bar = None

    @evaluator.on(Events.EPOCH_STARTED)
    def reset_validation_progress_bar(engine):
        nonlocal validation_progress_bar
        validation_progress_bar = create_progress_bar()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def update_validation_progress_bar(engine):
        validation_progress_bar.update()

    @evaluator.on(Events.EPOCH_COMPLETED)
    @evaluator.on(Events.EXCEPTION_RAISED)
    def release_progress_bar(engine, *args):
        validation_progress_bar.close()
        nonlocal epoch
        epoch += 1
        
