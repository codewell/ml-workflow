import numpy as np


def log_examples(tag, trainer):
    def log_examples_(engine, logger, event_name):
        n_examples = min(5, len(engine.state.output['predictions']))
        indices = np.random.choice(
            len(engine.state.output['predictions']),
            n_examples,
            replace=False,
        )
        logger.writer.add_images(
            f'{tag}/predictions',
            np.stack([
                np.stack([np.array(
                    engine.state.output['predictions'][index]
                    .representation(
                        engine.state.output['examples'][index]
                    )
                )], axis=-1) / 255
                for index in indices
            ]),
            trainer.state.epoch,
            dataformats='NHWC',
        )
    return log_examples_
