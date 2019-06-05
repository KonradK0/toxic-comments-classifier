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


def _parse_queries():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the file with queries.')
    args = parser.parse_args()

    with open(args.file) as queries_file:
        return queries_file.read().splitlines(keepends=False)


def _make_input_examples(queries):
    return [bert.run_classifier.InputExample(guid=None, text_a=query) for query in queries]


def _prepare_payload(tokenizer, input_examples):
    input_ids, input_masks, segment_ids, _ = datasets.utils.convert_examples_to_features(tokenizer, input_examples)

    payload = {
        "instances": [{
            "input_ids": input_ids.tolist(),
            "input_masks": input_masks.tolist(),
            "segment_ids": segment_ids.tolist()
        }]
    }

    return payload


if __name__ == '__main__':
    queries = _parse_queries()
    config = _parse_config()
    endpoint = f'http://{config["host"]}:{config["port"]}/{config["endpoint"]}'
    input_examples = _make_input_examples(queries)
    tokenizer = bert.tokenization.FullTokenizer(VOCAB_FNAME)
    payload = _prepare_payload(tokenizer, input_examples)
    r = requests.post(endpoint, json=payload)
    print(r.content)
