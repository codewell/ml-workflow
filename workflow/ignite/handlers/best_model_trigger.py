from enum import Enum
from ignite.engine.engine import CallableEvents
from ignite.engine import Events
import numpy as np


class CustomEvents(CallableEvents, Enum):
    NEW_BEST_MODEL = 'new_best_model'


class BestModelTrigger:
    Event = CustomEvents.NEW_BEST_MODEL

    def __init__(self, metric_name, extra_triggered_engines=[]):
        self.best_score = -np.inf
        self.extra_triggered_engines = extra_triggered_engines
        self.metric_name = metric_name

        for engine in extra_triggered_engines:
            engine.register_events(
                CustomEvents.NEW_BEST_MODEL,
                event_to_attr={CustomEvents.NEW_BEST_MODEL: 'new_best_model'}
            )

    def _check_new_best_model(self, engine):
        score = engine.state.metrics[self.metric_name]
        if score > self.best_score:
            self.best_score = score

            engine.fire_event(CustomEvents.NEW_BEST_MODEL)

            for trigger_engine in self.extra_triggered_engines:
                trigger_engine.fire_event(CustomEvents.NEW_BEST_MODEL)

    def attach(self, engine):
        engine.register_events(
            CustomEvents.NEW_BEST_MODEL,
            event_to_attr={CustomEvents.NEW_BEST_MODEL: 'new_best_model'}
        )

        engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda engine: self._check_new_best_model(engine)
        )
