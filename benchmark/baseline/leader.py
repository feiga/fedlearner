# Copyright 2020 The FedLearner Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
# pylint: disable=no-else-return, inconsistent-return-statements

import logging
import tensorflow.compat.v1 as tf
from fedlearner import trainer
from slot_2_bucket_map import local_ads_ad_slot_2_bucket

_SLOT_2_IDX = {pair[0]:i for i, pair in enumerate(local_ads_ad_slot_2_bucket)}
_SLOT_2_BUCKET = local_ads_ad_slot_2_bucket

ROLE = 'leader'

parser = trainer.trainer_worker.create_argument_parser()
parser.add_argument('--batch-size', type=int, default=256,
                    help='Training batch size.')
args = parser.parse_args()



def input_fn(bridge, trainer_master=None):
    dataset = trainer.data.DataBlockLoader(
        args.batch_size, ROLE, bridge, trainer_master).make_dataset()

    def parse_fn(example):
        feature_map = {
            "slot_id_{0}".format(k): tf.VarLenFeature(tf.int64) \
                                     for k, _ in _SLOT_2_BUCKET
        }
        feature_map["example_id"] = tf.FixedLenFeature([], tf.string)
        feature_map["label"] = tf.FixedLenFeature([], tf.int64)
        features = tf.parse_example(example, features=feature_map)
        labels = {'y': features.pop('label')}
        return features, labels
    dataset = dataset.map(map_func=parse_fn,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(2)
    return dataset


def serving_input_receiver_fn():
    feature_map = {"slot_id_{0}".format(k): tf.VarLenFeature(tf.int64) for k, _ in _SLOT_2_BUCKET}
    feature_map["example_id"] = tf.FixedLenFeature([], tf.string)

    record_batch = tf.placeholder(dtype=tf.string, name='examples')
    features = tf.parse_example(record_batch, features=feature_map)
    features['user_embed'] = tf.placeholder(dtype=tf.float32, name='user_embed')
    receiver_tensors = {
        'examples': record_batch,
        'user_embed': features['user_embed']
    }
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def model_fn(model, features, labels, mode):
    """Model Builder of wide&deep learning models
    Args:
    Returns
    """
    global_step = tf.train.get_or_create_global_step()
    num_slot, embed_size = len(_SLOT_2_BUCKET), 4
    xavier_initializer = tf.glorot_normal_initializer()

    columns = [tf.feature_column.categorical_column_with_hash_bucket(
                        key='slot_id_{0}'.format(slot),
                        hash_bucket_size=bucket_size,
                        dtype=tf.int64)
                    for slot, bucket_size in _SLOT_2_BUCKET]
    embed_output = [tf.feature_column.embedding_column(
                        col,
                        dimension=embed_size,
                        combiner='sum',
                        initializer=xavier_initializer)
                    for col in columns]
    concat_embedding = tf.feature_column.input_layer(
        features,
        feature_columns=embed_output)
    output_size = len(columns) * embed_size

    output_size = num_slot * embed_size
    fc1_size, fc2_size, fc3_size = 512, 256, 64
    w1 = tf.get_variable('w1', shape=[output_size, fc1_size], dtype=tf.float32,
                          initializer=xavier_initializer)
    b1 = tf.get_variable(
        'b1', shape=[fc1_size], dtype=tf.float32, initializer=tf.zeros_initializer())
    w2 = tf.get_variable('w2', shape=[fc1_size, fc2_size], dtype=tf.float32,
                         initializer=xavier_initializer)
    b2 = tf.get_variable(
        'b2', shape=[fc2_size], dtype=tf.float32, initializer=tf.zeros_initializer())
    w3 = tf.get_variable('w3', shape=[fc2_size, fc3_size], dtype=tf.float32,
                         initializer=xavier_initializer)
    b3 = tf.get_variable(
        'b3', shape=[fc3_size], dtype=tf.float32, initializer=tf.zeros_initializer())

    act1_l = tf.nn.relu(tf.nn.bias_add(tf.matmul(concat_embedding, w1), b1))
    act2_l = tf.nn.relu(tf.nn.bias_add(tf.matmul(act1_l, w2), b2))
    ad_embed = tf.nn.bias_add(tf.matmul(act2_l, w3), b3)

    if mode == tf.estimator.ModeKeys.TRAIN:
        user_embed = model.recv('user_embed', tf.float32, require_grad=True)
    else:
        user_embed = features['user_embed']


    logits = tf.reduce_sum(tf.multiply(user_embed, ad_embed), axis=1)
    logging.info("last_layer_concat value False")



    if mode == tf.estimator.ModeKeys.TRAIN:
        y = labels['y']

        y = tf.dtypes.cast(y, tf.float32)
        cross_entropy_fix = tf.log(2 + tf.exp(-logits)) + tf.cast((y - 1), tf.float32) * tf.log(1 + tf.exp(-logits))
        loss = tf.math.reduce_mean(cross_entropy_fix)

        # cala auc
        pred = 1 / (2 + (tf.exp(-logits)))
        _, auc = tf.metrics.auc(labels=y, predictions=pred)


        logging_hook = tf.train.LoggingTensorHook(
            {"loss" : loss, "auc": auc}, every_n_iter=10)

        step_counter_hook = tf.train.StepCounterHook(every_n_steps=100)

        optimizer = tf.train.GradientDescentOptimizer(0.1)
        # optimizer = tf.train.FtrlOptimizer(learning_rate=0.16921544485102483,
        #     l1_regularization_strength=1e-05, l2_regularization_strength=0.0005945795938393141,
        #     initial_accumulator_value=0.44352,
        #     learning_rate_power=-0.59496)
        train_op = model.minimize(optimizer, loss, global_step=global_step)
        return model.make_spec(mode, loss=loss, train_op=train_op,
                               training_hooks=[logging_hook, step_counter_hook])
    if mode == tf.estimator.ModeKeys.PREDICT:
        prob = 1 / (1 + (tf.exp(-logits)))
        output = {'prob': prob, 'ad_embed': ad_embed}
        return model.make_spec(mode, predictions=output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    trainer.trainer_worker.train(
        ROLE, args, input_fn,
        model_fn, serving_input_receiver_fn)

