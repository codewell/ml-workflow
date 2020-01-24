import os
import json


def split_new_data(df, key_column, split_fn, file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            previous_split = json.load(file)

        df = df[~df[key_column].isin(sum(previous_split, []))]

        if len(df) == 0:
            raise Exception('No new data found')

        split_dfs = split_fn(df)
        split = [
            df[key_column].tolist() + previous
            for df, previous in zip(split_dfs, previous_split)
        ]

    else:
        split_dfs = split_fn(df)
        split = [
            df[key_column].tolist()
            for df in split_dfs
        ]

    with open(file_path, 'w') as file:
        json.dump(split, file)

    return split
