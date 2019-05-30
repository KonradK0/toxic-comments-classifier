import os.path

import pandas as pd

import datasets.utils

DATASET_NAME = 'quora'


def _remove_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].str.replace(r'\s+', ' ')
    df['text'] = df['text'].str.strip()

    return df


def _get_preprocessed_dataset(*, raw_fname: str) -> pd.DataFrame:
    raw_path = os.path.join(datasets.utils.RAW_DIR, raw_fname)

    dataset = pd.read_csv(raw_path, usecols=['question_text', 'target'])
    dataset.rename(columns={'question_text': 'text', 'target': 'toxicity'}, inplace=True)
    dataset = _remove_tokens(dataset)
    dataset['text'] = dataset['text'].str.lower()

    return dataset


def fetch(fname: str = 'quora.csv', raw_fname: str = 'quora_raw.csv') -> pd.DataFrame:
    return datasets.utils.fetch(DATASET_NAME, fname, preprocess_fn=_get_preprocessed_dataset, raw_fname=raw_fname)
