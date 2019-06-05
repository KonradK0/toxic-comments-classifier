import tensorflow as tf
import tensorflow_hub as hub

BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_path=BERT_PATH, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.bert_path = bert_path
        self.trainable = True
        self.output_size = 768
        self.kwargs = kwargs
        super().__init__(**self.kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if "/cls/" not in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super().build(input_shape)

    def call(self, inputs):
        inputs = [tf.keras.backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size

    def get_config(self):
        config = super().get_config()
        config['bert_path'] = self.bert_path
        config['n_fine_tune_layers'] = self.n_fine_tune_layers
        return config
