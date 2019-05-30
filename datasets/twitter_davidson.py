import logging
import os.path

import pandas as pd

logger = logging.getLogger(__name__)
DATA_DIR = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else os.path.join('..', 'data')
DATASET_NAME = 'twitter_davidson'


def _remove_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].str.replace(r'\s+', ' ')
    df['text'] = df['text'].str.strip()

    return df


def _binarize_classes(df: pd.DataFrame) -> pd.DataFrame:
    toxicity = df['class'] < 2
    df['toxicity'] = toxicity.astype(int)

    return df


def _get_preprocessed_dataset(raw_fname: str):
    raw_path = os.path.join(DATA_DIR, DATASET_NAME, raw_fname)

    dataset = pd.read_csv(raw_path, usecols=['tweet', 'class'])
    dataset.rename(columns={'tweet': 'text'}, inplace=True)
    dataset = _binarize_classes(dataset)
    dataset = _remove_tokens(dataset)
    dataset['text'] = dataset['text'].str.lower()
    dataset = dataset[['text', 'toxicity']]

    return dataset


def fetch(fname: str = 'twitter_davidson.csv', raw_fname: str = 'davidson_raw.csv') -> pd.DataFrame:
    dataset_path = os.path.join(DATA_DIR, DATASET_NAME, fname)

    if not os.path.exists(dataset_path):
        logger.info(f'{fname} not present. Processing the dataset...')

        dataset = _get_preprocessed_dataset(raw_fname)
        dataset.to_csv(dataset_path, index=False)

        logger.info(f'{DATASET_NAME} processed')
    else:
        dataset = pd.read_csv(dataset_path, dtype={'text': str, 'toxicity': int})

    positive, negative = dataset['toxicity'].value_counts().T.to_numpy()
    logger.info(f'{DATASET_NAME} dataset: {positive} non-toxic, {negative} toxic')

    return dataset
