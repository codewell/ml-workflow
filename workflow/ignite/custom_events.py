from enum import Enum
from ignite.engine.engine import CallableEvents


class CustomEvents(CallableEvents, Enum):
    NEW_BEST_MODEL = 'new_best_model'
