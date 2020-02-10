

def warmup(n_steps):
    def _warmup(step, multiplier):
        if step == 0:
            return (0, 0.)
        elif step <= n_steps:
            return (1, (step - 1) / n_steps)
        else:
            return (step - n_steps, multiplier)
    return _warmup
