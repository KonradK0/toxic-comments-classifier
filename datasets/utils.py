import logging
import os.path
from typing import Any, Tuple

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer

logger = logging.getLogger(__name__)

DATA_DIR = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else os.path.join('..', 'data')
RAW_DIR = os.environ['RAW_DIR'] if 'RAW_DIR' in os.environ else os.path.join('..', 'data', 'raw')
PREPROCESSED_DIR = os.environ['PREPROCESSED_DIR'] if 'PREPROCESSED_DIR' in os.environ else os.path.join('..', 'data',
                                                                                                        'preprocessed')
DATASET_FILENAME = 'toxic_comments.csv'


def fetch(dataset_name: str, out_fname: str, *, preprocess_fn: [[..., Any], pd.DataFrame],
          **preprocess_kwargs) -> pd.DataFrame:
    dataset_path = os.path.join(PREPROCESSED_DIR, out_fname)

    if not os.path.exists(dataset_path):
        logger.info(f'{out_fname} not present. Processing the dataset...')

        dataset = preprocess_fn(**preprocess_kwargs)
        dataset.to_csv(dataset_path, index=False)

        logger.info(f'{dataset_name} processed')
    else:
        dataset = pd.read_csv(dataset_path, dtype={'text': str, 'toxicity': int})

    positive, negative = dataset['toxicity'].value_counts().T.to_numpy()
    logger.info(f'{dataset_name} dataset: {positive} non-toxic, {negative} toxic')

    return dataset


def _merge_datasets(file_extension: str) -> pd.DataFrame:
    paths = [os.path.join(PREPROCESSED_DIR, file) for file in os.listdir(path=PREPROCESSED_DIR) if
             file.endswith(file_extension)]
    dataset = pd.concat(pd.read_csv(path) for path in paths)
    return dataset


def get_toxic_comments_df(file_extension: str = 'csv', out_fname: str = DATASET_FILENAME) -> pd.DataFrame:
    out_path = os.path.join(DATA_DIR, out_fname)
    if not os.path.exists(out_path):
        dataset = _merge_datasets(file_extension)
        dataset.to_csv(out_path, index=False)
    else:
        dataset = pd.read_csv(out_path)
    return dataset.dropna()


def train_test_split(dataset: pd.DataFrame, test_ratio: float = 0.2) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = dataset.sample(frac=1)
    size = len(dataset)
    train_size = int((1 - test_ratio) * size)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    return train_dataset, test_dataset
