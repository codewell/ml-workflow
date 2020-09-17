import numpy as np
from datastream import Datastream

from {{cookiecutter.package_name}}.datastream import (
    evaluate_datastreams, augmenter
)
from {{cookiecutter.package_name}}.problem import settings


def GradientDatastream():
    dataset = evaluate_datastreams()['gradient'].dataset
    
    augmenter_ = augmenter()
    return (
        Datastream.merge([
            Datastream(dataset.subset(
                lambda df: df['class_name'] == class_name
            ))
            for class_name in settings.CLASS_NAMES
        ])
        .map(lambda example: example.augmented(augmenter_))
    )
