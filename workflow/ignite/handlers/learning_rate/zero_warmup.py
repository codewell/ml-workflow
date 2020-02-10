

def zero_warmup(n_steps):
    def _zero_warmup(step, multiplier):
        if step <= n_steps:
            return (1, 0.)
        else:
            return (step - n_steps, multiplier)
    return _zero_warmup
