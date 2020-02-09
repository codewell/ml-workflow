

def decay(base):
    def _decay(step, learning_rate):
        return (
            step,
            learning_rate * base ** step
        )
    return _decay
