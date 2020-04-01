

def cyclical(length=100, relative_min=0.1):
    '''Also known as triangular, https://arxiv.org/pdf/1506.01186.pdf'''

    half = length / 2
    def _cyclical(step, multiplier):
        return (
            step,
            multiplier * (
                relative_min + (1 - relative_min) * abs(
                    ((step - 1) % length - half)
                    / (half - 1)
                )
            )
        )
    return _cyclical
