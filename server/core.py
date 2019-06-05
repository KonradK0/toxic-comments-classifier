import os.path

import flask

from server.model import Classifier

PRETRAINED_PATH = os.environ['PRETRAINED_PATH'] if 'PRETRAINED_PATH' in os.environ else os.path.join('..', 'pretrained',
                                                                                                     'model.h5')
app = flask.Flask(__name__)


def get_model():
    model = getattr(flask.g, '_model', None)

    if model is None:
        model = flask.g._model = Classifier(PRETRAINED_PATH)

    return model


@app.route('/predict', methods=['POST'])
def predict():
    request = flask.request.get_json(silent=True)
    input_ids, input_masks, segment_ids = request['input_ids'], request['input_masks'], request['segment_ids']
    model = get_model()
    preds = model.predict(input_ids, input_masks, segment_ids)
    response = {'predictions': preds}

    return flask.jsonify(response)


if __name__ == '__main__':
    app.run(port=8081)
