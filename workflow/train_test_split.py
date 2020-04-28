from pathlib import Path
import numpy as np
import pandas as pd

from workflow import json


def load_split(split_file):
    if split_file.exists():
        saved_split = json.read(split_file)
    else:
        saved_split = dict(train=[], test=[])
    return saved_split


def train_test_split(dataframe, key, test_size, stratify, split_file):
    '''
    Continue splitting new examples in the background. Use a symbolic link to
    the split file or directory when using guild so the split is global.

    As new examples come in it can handle:
    - Changing test size
    - Adapt after removing examples from dataset
    - Adapt to new stratification

    Compared to hash splitting it selects a determnistic number of samples
    which is preferred when there is a small number of samples per strata.
    '''

    split_dataframe = dataframe[[key]].copy()
    if type(stratify) is str:
        stratify = dataframe[stratify]
    else:
        stratify = pd.Series(stratify)
    split_file = Path(split_file)

    saved_split = load_split(split_file)

    split_dataframe['split'] = 'unassigned'
    for split, keys in saved_split.items():
        split_dataframe.loc[lambda df: df[key].isin(keys), 'split'] = split

    for value, count in stratify.value_counts().iteritems():

        strata = split_dataframe[stratify == value]
        saved_n_test = (strata['split'] == 'test').sum()

        unassigned = (
            strata[lambda df: df['split'] == 'unassigned']
        )

        if len(unassigned) >= 1:

            float_test = count * test_size
            n_new_test = int(float_test) - saved_n_test

            probability = float_test - n_new_test
            if probability >= 1e-6 and np.random.rand() <= probability:
                n_new_test += 1

            if n_new_test >= 1:

                new_test = np.random.choice(
                    unassigned[key], size=n_new_test, replace=False
                )

                saved_split['test'] += list(map(str, new_test))
                saved_split['train'] += list(map(str, (
                    unassigned[key]
                    [lambda series: ~series.isin(new_test)]
                )))

            else:
                saved_split['train'] += list(map(str, unassigned[key]))

    json.write(saved_split, split_file)

    test_mask = dataframe[key].isin(saved_split['test'])
    return (
        dataframe[~test_mask],
        dataframe[test_mask],
    )


def test_train_test_split():
    pass
    # dataframe = pd.DataFrame(dict(
    #     key=range(10),
    #     class_names=np.int32(np.random.randn(10) >= 2),
    # ))

    # split_file = 'split_file.json'
    # key = 'key'
    # test_size = 0.4
    # stratify = dataframe['class_names']

    # train, test = train_test_split(dataframe, key, test_size, stratify, split_file)

    # print(train)
    # print(test)
