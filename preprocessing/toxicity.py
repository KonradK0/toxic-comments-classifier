import os.path

import pandas as pd

DATA_DIR = os.path.join('..', 'data', 'toxicity')


def join_datasets(*, comments_fname: str, annotations_fname: str) -> pd.DataFrame:
    dtypes = {'rev_id': int, 'toxicity': int, 'comment': str}

    comments_path = os.path.join(DATA_DIR, comments_fname)
    annotations_path = os.path.join(DATA_DIR, annotations_fname)

    comments = pd.read_csv(comments_path, sep='\t', usecols=['rev_id', 'comment'], dtype=dtypes)
    annotations = pd.read_csv(annotations_path, sep='\t', usecols=['rev_id', 'toxicity'], dtype=dtypes)

    return pd.merge(comments, annotations, on='rev_id').groupby('rev_id')['toxicity'].agg(pd.Series.mode).to_frame()


if __name__ == '__main__':
    join_datasets(comments_fname='toxicity_annotated_comments.tsv', annotations_fname='toxicity_annotations.tsv')
