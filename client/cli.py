import argparse

import requests
import yaml

CONFIG_FNAME = 'heroku_config.yml'


def _parse_config():
    with open(CONFIG_FNAME) as config:
        return yaml.safe_load(config)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the file with queries.')
    parser.add_argument('--pretty-print', action='store_true', help='Prints queries and labels together. '
                                                                    'Otherwise, only labels are printed.')
    return parser.parse_args()


def _parse_queries(args):
    with open(args.file) as queries_file:
        return queries_file.read().splitlines(keepends=False)


def _fetch_predictions(queries, endpoint):
    payload = {'queries': queries}
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
