import logging
import os.path

import pandas as pd

logger = logging.getLogger(__name__)
DATA_DIR = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else os.path.join('..', 'data')
DATASET_NAME = 'wikipedia_comments'


def _join_text_and_scores(comments_fname: str, annotations_fname: str) -> pd.DataFrame:
    dtypes = {'rev_id': int, 'toxicity': int, 'comment': str}

    comments_path = os.path.join(DATA_DIR, DATASET_NAME, comments_fname)
    annotations_path = os.path.join(DATA_DIR, DATASET_NAME, annotations_fname)

    comments = pd.read_csv(comments_path, sep='\t', usecols=['rev_id', 'comment'], dtype=dtypes)
    annotations = pd.read_csv(annotations_path, sep='\t', usecols=['rev_id', 'toxicity'], dtype=dtypes)

    joined = pd.merge(comments, annotations, on='rev_id').groupby(['comment'])['toxicity'] \
        .agg(pd.Series.mode).to_frame()
    joined = joined.loc[joined['toxicity'].astype(str).isin(['0', '1'])]
    joined.index.names = ['text']
    joined.reset_index(level=0, inplace=True)

    return joined


def _remove_tokens(df: pd.DataFrame, newline_token: str = 'NEWLINE_TOKEN',
                   tab_token: str = 'TAB_TOKEN') -> pd.DataFrame:
    df['text'] = df['text'].str.replace(rf'({newline_token}|{tab_token}|\s+)', ' ')
    df['text'] = df['text'].str.strip()

    return df


def _get_preprocessed_dataset(annotations_fname: str, comments_fname: str):
    dataset = _join_text_and_scores(comments_fname, annotations_fname)
    dataset = _remove_tokens(dataset)
    dataset['text'] = dataset['text'].str.lower()

    return dataset


def fetch(fname: str = 'wikipedia_comments.csv', comments_fname: str = 'toxicity_annotated_comments.tsv',
          annotations_fname: str = 'toxicity_annotations.tsv') -> pd.DataFrame:
    dataset_path = os.path.join(DATA_DIR, DATASET_NAME, fname)

    if not os.path.exists(dataset_path):
        logger.info(f'{fname} not present. Processing the dataset...')

        dataset = _get_preprocessed_dataset(annotations_fname, comments_fname)
        dataset.to_csv(dataset_path, index=False)

        logger.info(f'{DATASET_NAME} processed')
    else:
        dataset = pd.read_csv(dataset_path, dtype={'text': str, 'toxicity': int})

    positive, negative = dataset['toxicity'].value_counts().T.to_numpy()
    logger.info(f'Wikipedia comments dataset: {positive} non-toxic, {negative} toxic')

    return dataset
