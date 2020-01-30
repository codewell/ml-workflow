import os
import json


def split_new_data(df, key_column, split_fn, filepath):
    '''
    Usage:

        split_fn = lambda df: sklearn.model_selection.train_test_split(
            df,
            stratify=df['source'],
            test_size=config['test_size'],
            random_state=config['seed'],
        )
        split_new_data(
            voice_df, 'image_filepath', split_fn, config['voice_split_path']
        )

    '''
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
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

    with open(filepath, 'w') as file:
        json.dump(split, file, indent=4)

    return split
