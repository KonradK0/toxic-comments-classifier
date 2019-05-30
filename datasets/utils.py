import logging
import os.path

import pandas as pd

logger = logging.getLogger(__name__)
DATA_DIR = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else os.path.join('..', 'data')


def fetch(dataset_name, out_fname, *, preprocess_fn, **preprocess_kwargs):
    dataset_path = os.path.join(DATA_DIR, dataset_name, out_fname)

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
