import numpy as np
from datastream import Datastream

from {{cookiecutter.package_name}}.data import datasets, augment
from {{cookiecutter.package_name}}.problem import settings


def GradientDatastream():
    dataset = datasets()['gradient']

    return (
        Datastream.merge([
            Datastream(
                dataset.subset(lambda df: df['class_name'] == class_name)
            )
            for class_name in settings.CLASS_NAMES
        ])
        .map(augment)
    )
