import numpy as np
from workflow.torch import Datastream

from {{cookiecutter.package_name}}.data import datasets, augment
from {{cookiecutter.package_name}}.problem import settings


def GradientDatastream():
    dataset = datasets()['gradient']

    return (
        Datastream.merge([
            Datastream(dataset.subset(np.argwhere(
                (dataset.source['class_name'].values == class_name)
            ).squeeze()))
            for class_name in settings.CLASS_NAMES
        ])
        .map(augment)
    )
