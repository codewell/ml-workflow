

def cyclical(length=100, relative_min=0.1):
    def _cyclical(step, learning_rate):
        return (
            step,
            learning_rate * (
                1 - (step % length) / length
            ) * (1- relative_min) + relative_min
        )
    return _cyclical
