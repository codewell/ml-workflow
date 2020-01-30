import math
from tqdm import tqdm
from ignite.engine import Events
from workflow.ignite.constants import TQDM_OUTFILE


def attach_validation_progress_bar(evaluator, n_batches, config):

    validation_progress_bar = tqdm(
        total=n_batches, file=TQDM_OUTFILE, desc='Validation'
    )

    @evaluator.on(Events.EPOCH_STARTED)
    def reset_validation_progress_bar(evaluator):
        validation_progress_bar.reset()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def update_validation_progress_bar(evaluator):
        validation_progress_bar.update()
