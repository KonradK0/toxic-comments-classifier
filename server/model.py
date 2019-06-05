import tensorflow as tf

import layers.custom


class Classifier:
    def __init__(self, pretrained_path):
        self.model = tf.keras.models.load_model(pretrained_path, custom_objects={'BertLayer': layers.custom.BertLayer})
        self.graph = tf.get_default_graph()

    def predict(self, *inputs):
        with self.graph.as_default():
            probabilities = self.model.predict(inputs).flatten()
            predictions = (probabilities > 0.5).astype(int).tolist()
            return predictions
