import ignite


class ProgressBar(ignite.contrib.handlers.tqdm_logger.ProgressBar):
    def __init__(self, desc, *args, **kwargs):
        kwargs['smoothing'] = kwargs.get('smoothing', .0)
        super().__init__(
            *args,
            **kwargs,
            desc=desc,
            bar_format=(
                '{desc} {percentage:3.0f}%|{bar} {n}/{total}{postfix} '
                '[{elapsed}<{remaining} {rate_fmt}]'
            ),
        )
