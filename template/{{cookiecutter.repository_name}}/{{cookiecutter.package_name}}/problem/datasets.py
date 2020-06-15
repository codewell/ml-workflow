import pandas as pd
import torchvision
from datastream import Dataset


def datasets():
    train_dataset = torchvision.datasets.MNIST(
        'cache', train=True, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        'cache', train=False, download=True
    )

    return dict(
        train=(
            Dataset.from_dataframe(
                pd.DataFrame(dict(
                    index=range(len(train_dataset)),
                    class_name=train_dataset.targets,
                ))
            )
            .map(lambda row: row['index'])
            .map(lambda index: train_dataset[index])
            .map(lambda image, class_name: dict(
                image=image,
                class_name=class_name,
            ))
        ),
        compare=(
            Dataset.from_subscriptable(list(range(len(test_dataset))))
            .map(lambda index: test_dataset[index])
            .map(lambda image, class_name: dict(
                image=image,
                class_name=class_name,
            ))
        ),
    )
