import logging
import os.path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)
DATA_DIR = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else os.path.join('..', 'data')
DATASET_NAME = 'wikipedia_jigsaw'


def _remove_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].str.replace(r'\s+', ' ')
    df['text'] = df['text'].str.strip()

    return df


def _merge_toxicity_labels(df: pd.DataFrame, toxicity_criteria: List[str]) -> pd.DataFrame:
    toxicity = df[(df.columns & toxicity_criteria).any()]
    df['toxicity'] = toxicity

    return df


def _get_preprocessed_dataset(raw_fname: str):
    raw_path = os.path.join(DATA_DIR, DATASET_NAME, raw_fname)
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
    dataset_path = os.path.join(DATA_DIR, DATASET_NAME, fname)

    if not os.path.exists(dataset_path):
        logger.info(f'{fname} not present. Processing the dataset...')

        dataset = _get_preprocessed_dataset(raw_fname)
        dataset.to_csv(dataset_path, index=False)

        logger.info(f'{DATASET_NAME} processed')
    else:
        dataset = pd.read_csv(dataset_path, dtype={'text': str, 'toxicity': int})

    positive, negative = dataset['toxicity'].value_counts().T.to_numpy()
    logger.info(f'Wikipedia jigsaw dataset: {positive} non-toxic, {negative} toxic')

    return dataset


if __name__ == '__main__':
    jigsaw = fetch()
