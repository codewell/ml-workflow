

def cyclical(length=100, relative_min=0.1):
    def _cyclical(step, multiplier):
        return (
            step,
            multiplier * (
                1 - (1 - relative_min) * ((step - 1) % length) / (length - 1)
            )
        )
    return _cyclical
