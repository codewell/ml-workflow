import numpy as np


def loss_score_function(engine):
    score = -engine.state.metrics['loss']
    return -np.inf if score == 0 else score
