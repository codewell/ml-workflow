import numpy as np

from mnist.problem import settings


def preprocess(example):
    return dict(
        image=np.expand_dims(
            np.array(example['image'], dtype=np.float32),
            axis=0,
        ),
        class_name=example['class_name'],
    )

