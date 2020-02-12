from ignite.engine import Events
from workflow.ignite.custom_events import CustomEvents
import numpy as np


class BestModelTracker:
    def __init__(self):
        self.best_score = -np.inf

    def attach(self, engine, model_score_function):
        def check_if_new_best_model(engine):
            score = model_score_function(engine)
            if score > self.best_score:
                self.best_score = score
                engine.fire_event(CustomEvents.NEW_BEST_MODEL)

        engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda engine: check_if_new_best_model(engine)
        )
