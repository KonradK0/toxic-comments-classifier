import os

import tensorflow as tf
from bert import run_classifier, modeling
from bert.tokenization import FullTokenizer

import datasets

BERT_DIR = os.environ['BERT_DIR'] if 'BERT_DIR' in os.environ else os.path.join('..', 'pretrained', 'bert')
BERT_CONFIG = os.path.join(BERT_DIR, 'bert_config.json')
LEARNING_RATE = os.environ['LEARNING_RATE'] if 'LEARNING_RATE' in os.environ else 0.005
OUTPUT_DIR = os.environ['OUTPUT_DIR'] if 'OUTPUT_DIR' in os.environ else os.path.join('..', 'out')

SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100
BATCH_SIZE = 1

num_train_steps = 1
num_warmup_steps = 1

dataset = datasets.utils.get_toxic_comments_df()[:100]
train, test = datasets.utils.train_test_split(dataset)

train_InputExamples = train.apply(
    lambda x: run_classifier.InputExample(guid=None,
                                          text_a=x['text'],
                                          text_b=None,
                                          label=x['toxicity']), axis=1)

test_InputExamples = test.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                      text_a=x['text'],
                                                                      text_b=None,
                                                                      label=x['toxicity']), axis=1)

label_list = [0, 1]
max_seq_len = dataset['text'].map(lambda x: len(x)).max()
tokenizer = FullTokenizer(vocab_file=os.path.join(BERT_DIR, 'vocab.txt'), do_lower_case=True)

train_features = run_classifier.convert_examples_to_features(train_InputExamples, label_list, max_seq_len, tokenizer)
test_features = run_classifier.convert_examples_to_features(test_InputExamples, label_list, max_seq_len, tokenizer)

model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(BERT_CONFIG),
    num_labels=2,
    init_checkpoint=BERT_DIR,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False,
    use_one_hot_embeddings=False)

run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={"batch_size": BATCH_SIZE})

train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=max_seq_len,
    is_training=True,
    drop_remainder=False)

test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=max_seq_len,
    is_training=False,
    drop_remainder=False)

print(estimator.evaluate(input_fn=test_input_fn, steps=None))
