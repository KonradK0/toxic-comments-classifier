import logging
import os.path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='')

DATA_DIR = os.path.join('..', 'data', 'toxicity')


def join_datasets(*, comments_fname: str, annotations_fname: str) -> pd.DataFrame:
    dtypes = {'rev_id': int, 'toxicity': int, 'comment': str}

    comments_path = os.path.join(DATA_DIR, comments_fname)
    annotations_path = os.path.join(DATA_DIR, annotations_fname)

    comments = pd.read_csv(comments_path, sep='\t', usecols=['rev_id', 'comment'], dtype=dtypes)
    annotations = pd.read_csv(annotations_path, sep='\t', usecols=['rev_id', 'toxicity'], dtype=dtypes)

    joined = pd.merge(comments, annotations, on='rev_id').groupby('rev_id')['toxicity'].agg(
        pd.Series.mode).to_frame()
    joined = joined.loc[joined['toxicity'].astype(str).isin(['0', '1'])]
    return joined.astype(int)


def get_toxicity_df(fpath):
    if not os.path.exists(fpath):
        joined = join_datasets(comments_fname='toxicity_annotated_comments.tsv',
                               annotations_fname='toxicity_annotations.tsv')
        joined.to_csv(fpath)
    else:
        joined = pd.read_csv(fpath, dtype={'rev_id': int, 'toxicity': int})
    return joined


if __name__ == '__main__':
    fname = 'toxicity.csv'
    joined = get_toxicity_df(fpath=os.path.join(DATA_DIR, fname))
    logging.info(joined['toxicity'].value_counts())
