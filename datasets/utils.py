import logging
import os.path
from typing import Any

import bert.run_classifier
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else os.path.join('..', 'data')
RAW_DIR = os.environ['RAW_DIR'] if 'RAW_DIR' in os.environ else os.path.join(DATA_DIR, 'raw')
PREPROCESSED_DIR = os.environ['PREPROCESSED_DIR'] if 'PREPROCESSED_DIR' in os.environ else os.path.join(DATA_DIR,
                                                                                                        'preprocessed')
DATASET_FILENAME = 'toxic_comments.csv'
MAX_SEQ_LEN = 128


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


def convert_single_example(tokenizer, example, max_seq_length=MAX_SEQ_LEN):
    if isinstance(example, bert.run_classifier.PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=MAX_SEQ_LEN):
    input_ids, input_masks, segment_ids, labels = [], [], [], []

    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )
