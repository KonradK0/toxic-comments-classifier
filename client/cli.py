import argparse

import bert.run_classifier
import bert.tokenization
import requests
import yaml

import datasets

CONFIG_FNAME = 'test_config.yml'
VOCAB_FNAME = 'vocab.txt'


def _parse_config():
    with open(CONFIG_FNAME) as config:
        return yaml.safe_load(config)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the file with queries.')
    parser.add_argument('--pretty-print', action='store_true', help='Prints queries and labels together. '
                                                                    'Otherwise, only labels are printed.')
    return parser.parse_args()


def _make_input_examples(queries):
    return [bert.run_classifier.InputExample(guid=None, text_a=query) for query in queries]


def _prepare_payload(tokenizer, input_examples):
    input_ids, input_masks, segment_ids, _ = datasets.utils.convert_examples_to_features(tokenizer, input_examples)

    payload = {
        "input_ids": input_ids.tolist(),
        "input_masks": input_masks.tolist(),
        "segment_ids": segment_ids.tolist()
    }

    return payload


def _parse_queries(args):
    with open(args.file) as queries_file:
        return queries_file.read().splitlines(keepends=False)


def _fetch_predictions(queries, endpoint):
    input_examples = _make_input_examples(queries)
    tokenizer = bert.tokenization.FullTokenizer(VOCAB_FNAME)
    payload = _prepare_payload(tokenizer, input_examples)
    r = requests.post(endpoint, json=payload)
    return r.json()['predictions']


def _pretty_print(queries, predictions):
    longest_query = max(len(query) for query in queries)

    for query, pred in zip(queries, predictions):
        print(f'{query.ljust(longest_query)} | class: {pred}')


def _label_print(predictions):
    for pred in predictions:
        print(pred)


if __name__ == '__main__':
    args = _parse_args()
    queries = _parse_queries(args)
    config = _parse_config()
    endpoint = f'http://{config["host"]}:{config["port"]}/{config["endpoint"]}'
    predictions = _fetch_predictions(queries, endpoint)

    assert len(predictions) == len(queries)

    if args.pretty_print:
        _pretty_print(queries, predictions)
    else:
        _label_print(predictions)
