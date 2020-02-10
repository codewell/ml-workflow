

def warmup(n_steps):
    def _warmup(step, multiplier):
        if step <= n_steps:
            return (1, 0.0)
        else:
            return (step - n_steps, multiplier)
    return _warmup
