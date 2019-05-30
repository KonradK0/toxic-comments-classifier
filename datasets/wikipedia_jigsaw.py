import os.path
from typing import List

import pandas as pd

import datasets.utils

DATASET_NAME = 'wikipedia_jigsaw'


def _remove_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].str.replace(r'\s+', ' ')
    df['text'] = df['text'].str.strip()

    return df


def _merge_toxicity_labels(df: pd.DataFrame, toxicity_criteria: List[str]) -> pd.DataFrame:
    toxicity = df[(df.columns & toxicity_criteria).any()]
    df['toxicity'] = toxicity

    return df


def _get_preprocessed_dataset(raw_fname: str) -> pd.DataFrame:
    raw_path = os.path.join(datasets.utils.RAW_DIR, raw_fname)
    toxicity_criteria = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    dtypes = {criterion: bool for criterion in toxicity_criteria}

    dataset = pd.read_csv(raw_path, dtype=dtypes, usecols=['comment_text'] + toxicity_criteria)
    dataset.rename(columns={'comment_text': 'text'}, inplace=True)
    dataset = _merge_toxicity_labels(dataset, toxicity_criteria)
    dataset = _remove_tokens(dataset)
    dataset['text'] = dataset['text'].str.lower()
    dataset['toxicity'] = dataset['toxicity'].astype(int)
    dataset = dataset[['text', 'toxicity']]

    return dataset


def fetch(fname: str = 'wikipedia_jigsaw.csv', raw_fname: str = 'jigsaw_raw.csv') -> pd.DataFrame:
    return datasets.utils.fetch(DATASET_NAME, fname, preprocess_fn=_get_preprocessed_dataset, raw_fname=raw_fname)
