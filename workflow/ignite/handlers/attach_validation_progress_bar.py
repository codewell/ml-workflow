import math
from tqdm import tqdm
from ignite.engine import Events
from workflow.ignite.constants import TQDM_OUTFILE


def attach_validation_progress_bar(evaluator, validation_loader, config):

    num_validation_steps = math.ceil(
        len(validation_loader.dataset) / config['eval_batch_size']
    )

    validation_progress_bar = tqdm(
        total=num_validation_steps, file=TQDM_OUTFILE, desc='Validation'
    )

    @evaluator.on(Events.EPOCH_STARTED)
    def reset_validation_progress_bar(evaluator):
        validation_progress_bar.reset()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def update_validation_progress_bar(evaluator):
        validation_progress_bar.update()
