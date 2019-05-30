import os.path

import pandas as pd

import datasets.utils

DATASET_NAME = 'wikipedia_comments'


def _join_text_and_scores(comments_fname: str, annotations_fname: str) -> pd.DataFrame:
    dtypes = {'rev_id': int, 'toxicity': int, 'comment': str}

    comments_path = os.path.join(datasets.utils.DATA_DIR, DATASET_NAME, comments_fname)
    annotations_path = os.path.join(datasets.utils.DATA_DIR, DATASET_NAME, annotations_fname)

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


def _get_preprocessed_dataset(*, annotations_fname: str, comments_fname: str) -> pd.DataFrame:
    dataset = _join_text_and_scores(comments_fname, annotations_fname)
    dataset = _remove_tokens(dataset)
    dataset['text'] = dataset['text'].str.lower()

    return dataset


def fetch(fname: str = 'wikipedia_comments.csv', comments_fname: str = 'toxicity_annotated_comments.tsv',
          annotations_fname: str = 'toxicity_annotations.tsv') -> pd.DataFrame:
    return datasets.utils.fetch(DATASET_NAME, fname, preprocess_fn=_get_preprocessed_dataset,
                                comments_fname=comments_fname, annotations_fname=annotations_fname)
