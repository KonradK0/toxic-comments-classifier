import os.path

import pandas as pd

import datasets.utils

DATASET_NAME = 'twitter_davidson'


def _remove_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].str.replace(r'\s+', ' ')
    df['text'] = df['text'].str.strip()

    return df


def _binarize_classes(df: pd.DataFrame) -> pd.DataFrame:
    toxicity = df['class'] < 2
    df['toxicity'] = toxicity.astype(int)

    return df


def _get_preprocessed_dataset(*, raw_fname: str) -> pd.DataFrame:
    raw_path = os.path.join(datasets.utils.DATA_DIR, DATASET_NAME, raw_fname)

    dataset = pd.read_csv(raw_path, usecols=['tweet', 'class'])
    dataset.rename(columns={'tweet': 'text'}, inplace=True)
    dataset = _binarize_classes(dataset)
    dataset = _remove_tokens(dataset)
    dataset['text'] = dataset['text'].str.lower()
    dataset = dataset[['text', 'toxicity']]

    return dataset


def fetch(fname: str = 'twitter_davidson.csv', raw_fname: str = 'davidson_raw.csv') -> pd.DataFrame:
    return datasets.utils.fetch(DATASET_NAME, fname, preprocess_fn=_get_preprocessed_dataset, raw_fname=raw_fname)
