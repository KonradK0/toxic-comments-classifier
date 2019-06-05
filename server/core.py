import os.path

import bert
import bert.run_classifier
import bert.tokenization
import flask

import datasets
from server.model import Classifier

PRETRAINED_PATH = os.environ['PRETRAINED_PATH'] if 'PRETRAINED_PATH' in os.environ else os.path.join('..', 'pretrained',
                                                                                                     'model.h5')
PORT = os.environ['PORT'] if 'PORT' in os.environ else 8080
VOCAB_FILE = 'vocab.txt'
app = flask.Flask(__name__)


def get_model():
    model = getattr(flask.g, '_model', None)

    if model is None:
        model = flask.g._model = Classifier(PRETRAINED_PATH)

    return model


def _parse_request():
    queries = flask.request.get_json(silent=True)['queries']
    tokenizer = bert.tokenization.FullTokenizer(VOCAB_FILE)

    input_examples = [bert.run_classifier.InputExample(guid=None, text_a=query) for query in queries]

    return datasets.utils.convert_examples_to_features(tokenizer, input_examples)


@app.route('/predict', methods=['POST'])
def predict():
    input_ids, input_masks, segment_ids, _ = _parse_request()
    model = get_model()
    preds = model.predict(input_ids, input_masks, segment_ids)
    response = {'predictions': preds}

    return flask.jsonify(response)


if __name__ == '__main__':
    app.run(port=PORT)
