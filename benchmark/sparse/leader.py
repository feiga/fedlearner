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
import fedlearner.trainer as flt
from slot_2_bucket_map import local_ads_ad_slot_2_bucket

_SLOT_2_IDX = {pair[0]:i for i, pair in enumerate(local_ads_ad_slot_2_bucket)}
_SLOT_2_BUCKET = local_ads_ad_slot_2_bucket

ROLE = 'leader'

parser = flt.trainer_worker.create_argument_parser()
parser.add_argument('--batch-size', type=int, default=256,
                    help='Training batch size.')
args = parser.parse_args()


def input_fn(bridge, trainer_master=None):
    dataset = flt.data.DataBlockLoader(
        args.batch_size, ROLE, bridge, trainer_master).make_dataset()

    def parse_fn(example):
        feature_map = {"fids": tf.VarLenFeature(tf.int64)}
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
    feature_map = {"fids": tf.VarLenFeature(tf.int64)}
    feature_map["example_id"] = tf.FixedLenFeature([], tf.string)

    record_batch = tf.placeholder(dtype=tf.string, name='examples')
    features = tf.parse_example(record_batch, features=feature_map)
    features['act1_f'] = tf.placeholder(dtype=tf.float32, name='act1_f')
    receiver_tensors = {
        'examples': record_batch,
        'act1_f': features['act1_f']
    }
    return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensors)


def model_fn(model, features, labels, mode):
    """Model Builder of wide&deep learning models
    Args:
    Returns
    """
    global_step = tf.train.get_or_create_global_step()
    xavier_initializer = tf.glorot_normal_initializer()

    flt.feature.FeatureSlot.set_default_bias_initializer(
        tf.zeros_initializer())
    flt.feature.FeatureSlot.set_default_vec_initializer(
        tf.random_uniform_initializer(-0.0078125, 0.0078125))
    flt.feature.FeatureSlot.set_default_bias_optimizer(
        tf.train.FtrlOptimizer(learning_rate=0.01))
    flt.feature.FeatureSlot.set_default_vec_optimizer(
        tf.train.AdagradOptimizer(learning_rate=0.01))

    num_slot, embed_size = len(_SLOT_2_BUCKET), 4

    for slot_id, hash_size in _SLOT_2_BUCKET:
        fs = model.add_feature_slot(slot_id, hash_size)
        fc = model.add_feature_column(fs)
        fc.add_vector(embed_size)

    model.freeze_slots(features)

    embed_output = model.get_vec()

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

    act1_l = tf.nn.relu(tf.nn.bias_add(tf.matmul(embed_output, w1), b1))
    act2_l = tf.nn.relu(tf.nn.bias_add(tf.matmul(act1_l, w2), b2))
    act2_l = tf.nn.bias_add(tf.matmul(act2_l, w3), b3)

    if mode == tf.estimator.ModeKeys.TRAIN:
        act1_f = model.recv('act1_f', tf.float32, require_grad=True)
    else:
        act1_f = features['act1_f']
    logits = tf.reduce_sum(tf.multiply(act2_l, act1_f), axis=1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        y = labels['y']
        y = tf.dtypes.cast(y, tf.float32)
        print('==> logits: ', logits, ' y: ', y)
        cross_entropy_fix = tf.log(2 + tf.exp(-logits)) + tf.cast((y - 1), tf.float32) * tf.log(1 + tf.exp(-logits))
        loss = tf.math.reduce_mean(cross_entropy_fix)
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            # labels=y, logits=logits)
        # loss = tf.math.reduce_mean(loss)

        # cala auc
        pred = 1 / (2 + (tf.exp(-logits)))
        print('y: ', y, ' pred: ', pred)
        _, auc = tf.metrics.auc(labels=y, predictions=pred)

        logging_hook = tf.train.LoggingTensorHook(
            {"loss" : loss, 'auc': auc}, every_n_iter=10)

        step_counter_hook = tf.train.StepCounterHook(every_n_steps=1000)

        optimizer = tf.train.GradientDescentOptimizer(0.1)
        train_op = model.minimize(optimizer, loss, global_step=global_step)
        return model.make_spec(mode, loss=loss, train_op=train_op,
                               training_hooks=[logging_hook, step_counter_hook])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return model.make_spec(mode, predictions=logits)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    flt.trainer_worker.train(
        ROLE, args, input_fn,
        model_fn, serving_input_receiver_fn)

