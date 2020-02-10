

def decay(base):
    def _decay(step, multiplier):
        return (
            step,
            multiplier * base ** (step - 1)
        )
    return _decay
