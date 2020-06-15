from pathlib import Path
from PIL import Image
import pandas as pd
from datastream import Dataset

from {{cookiecutter.package_name}} import problem


def MnistDataset(dataframe):
    return (
        Dataset.from_dataframe(dataframe)
        .map(lambda row: (
            Path(row['image_path']),
            row['class_name'],
        ))
        .map(lambda image_path, class_name: dict(
            image=Image.open('prepare' / image_path),
            class_name=class_name,
        ))
    )


def datasets():
    train_df = pd.read_csv('prepare' / problem.settings.TRAIN_CSV)
    test_df = pd.read_csv('prepare' / problem.settings.TEST_CSV)

    return dict(
        train=MnistDataset(train_df),
        compare=MnistDataset(test_df),
    )
