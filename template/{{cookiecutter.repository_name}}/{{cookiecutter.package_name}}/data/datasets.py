import numpy as np
import pandas as pd
from workflow import train_test_split
from datastream import Dataset

from {{cookiecutter.package_name}}.problem import datasets as problem_datasets


def datasets():
    problem_datasets_ = problem_datasets()
    train_dataset = problem_datasets_['train']
    gradient, early_stopping = train_test_split(
        train_dataset.dataframe,
        key='index',
        test_size=0.2,
        stratify='class_name',
        split_file='{{cookiecutter.package_name}}/data/split.json',
    )

    return dict(
        gradient=train_dataset.subset(lambda df: (
            df['index'].isin(gradient['index'])
        )),
        early_stopping=train_dataset.subset(lambda df: (
            df['index'].isin(early_stopping['index'])
        )),
        compare=problem_datasets_['compare'],
    )
