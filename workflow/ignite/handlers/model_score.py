import ignite
from ignite.engine import Events
from ignite.contrib.handlers.tensorboard_logger import OutputHandler

from workflow.ignite.metrics import ReduceMetricsLambda
from workflow.ignite.handlers.early_stopping import EarlyStopping
from workflow.ignite.handlers.model_checkpoint import ModelCheckpoint
from workflow.ignite.handlers.best_model_trigger import BestModelTrigger


class ModelScore:
    def __init__(
        self,
        model_score_function,
        checkpoint_state,
        evaluator_metrics,
        tensorboard_logger,
        config,
    ):
        '''
        Model score brings together several handlers related to model score.
        - model checkpoint
        - early stopping
        - tensorboard logging
        - best model event
        '''

        self.model_score_function = model_score_function
        self.checkpoint_state = checkpoint_state
        self.evaluator_metrics = evaluator_metrics
        self.tensorboard_logger = tensorboard_logger
        self.config = config

    def attach(self, trainer, evaluators):
        def _model_score_function(*args, **kwargs):
            return self.model_score_function()

        ignite.metrics.MetricsLambda(_model_score_function).attach(trainer, 'model_score')
        ReduceMetricsLambda(max, _model_score_function).attach(trainer, 'best_model_score')

        training_desc = 'train'
        self.tensorboard_logger.attach(
            trainer,
            OutputHandler(
                tag=training_desc,
                metric_names=['model_score', 'best_model_score'],
            ),
            Events.EPOCH_COMPLETED,
        )

        BestModelTrigger('model_score', evaluators.values()).attach(trainer)

        for evaluator_desc, evaluator in evaluators.items():
            evaluator_metric_names = list(
                self.evaluator_metrics[evaluator_desc].keys()
            )
            self.tensorboard_logger.attach(
                evaluator,
                OutputHandler(
                    tag=f'best-{evaluator_desc}',
                    metric_names=evaluator_metric_names,
                    global_step_transform=lambda *args: trainer.state.epoch,
                ),
                BestModelTrigger.Event,
            )

        ModelCheckpoint(_model_score_function).attach(
            trainer, self.checkpoint_state
        )

        EarlyStopping(
            _model_score_function, trainer, self.config
        ).attach(trainer)
