import logging
import os.path

import pandas as pd

DATA_DIR = os.path.join('..', 'data', 'toxicity')
DATA_FILE = 'toxicity.csv'


def join_datasets(*, comments_fname: str, annotations_fname: str) -> pd.DataFrame:
    dtypes = {'rev_id': int, 'toxicity': int, 'comment': str}

    comments_path = os.path.join(DATA_DIR, comments_fname)
    annotations_path = os.path.join(DATA_DIR, annotations_fname)

    comments = pd.read_csv(comments_path, sep='\t', usecols=['rev_id', 'comment'], dtype=dtypes)
    annotations = pd.read_csv(annotations_path, sep='\t', usecols=['rev_id', 'toxicity'], dtype=dtypes)

    joined = pd.merge(comments, annotations, on='rev_id').groupby(by='comment')['toxicity'].agg(
        pd.Series.mode).to_frame()
    joined = joined.loc[joined['toxicity'].astype(str).isin(['0', '1'])]
    return joined.astype(int)


def get_toxicity_df(fpath: str) -> pd.DataFrame:
    if not os.path.exists(fpath):
        joined = join_datasets(comments_fname='toxicity_annotated_comments.tsv',
                               annotations_fname='toxicity_annotations.tsv')
        joined.to_csv(fpath)
    else:
        joined = pd.read_csv(fpath, dtype={'rev_id': int, 'toxicity': int})
    return joined


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    joined = get_toxicity_df(fpath=os.path.join(DATA_DIR, DATA_FILE))
    logger.info(joined['toxicity'].value_counts())
