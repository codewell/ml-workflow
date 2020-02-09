

def warmup(n_steps):
    def _warmup(step, learning_rate):
        if step <= n_steps:
            return (1, 0.0)
        else:
            return (step, learning_rate)
    return _warmup