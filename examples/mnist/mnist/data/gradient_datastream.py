import numpy as np
from workflow.torch import Datastream

from mnist.data import datasets, augment
from mnist.problem import settings


def GradientDatastream():
    dataset = datasets()['gradient']

    # from IPython import embed
    # embed()

    return (
        Datastream.merge([
            Datastream(dataset.subset(np.argwhere(
                (dataset.source['class_name'].values == class_name)
            ).squeeze()))
            for class_name in settings.CLASS_NAMES
        ])
        .map(augment)
    )
