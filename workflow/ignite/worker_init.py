from workflow.torch import set_seeds


def worker_init(seed, trainer, worker_id):
    set_seeds(
        seed * 2 ** 16
        + worker_id * 2 ** 24
        + trainer.state.epoch
    )
