from ignite.engine import Events
from workflow.ignite.custom_events import CustomEvents
import numpy as np


class BestModelTrigger:
    def __init__(self, engines_to_trigger, metric_to_track):
        self.best_score = -np.inf
        self.engines_to_trigger = engines_to_trigger
        self.metric_to_track = metric_to_track

    def _check_if_new_best_model(self, engine):
        score = engine.state.metrics[self.metric_to_track]
        if score > self.best_score:
            self.best_score = score
            for engine_to_trigger in self.engines_to_trigger:
                engine_to_trigger.fire_event(CustomEvents.NEW_BEST_MODEL)

    def attach(self, engine):
        engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda engine: self._check_if_new_best_model(engine)
        )
