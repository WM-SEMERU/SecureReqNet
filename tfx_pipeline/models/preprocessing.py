## Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX taxi preprocessing.

This file defines a template for TFX Transform component.
"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_transform as tft

from models import features

from securereqnet.utils import Embeddings

SEQUENCE_LENGTH = 100
VOCAB_SIZE = 10000
OOV_SIZE = 100

def _fill_in_missing(x, default_value):
  """Replace missing values in a SparseTensor.

  Fills in missing values of `x` with the default_value.

  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
    default_value: the value with which to replace the missing values.

  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)

def tokenize_issues(issues, sequence_length=SEQUENCE_LENGTH):
  issues = tf.strings.lower(issues)
  tokens = tf.strings.split(issues)[:, :sequence_length]
  start_tokens = tf.fill([tf.shape(issues)[0], 1], "<START>")
  end_tokens = tf.fill([tf.shape(issues)[0], 1], "<END>")
  tokens = tf.concat([start_tokens, tokens, end_tokens], axis=1)
  tokens = tokens[:, :sequence_length]
  tokens = tokens.to_tensor(default_value="<PAD>")
  pad = sequence_length - tf.shape(tokens)[1]
  tokens = tf.pad(tokens, [[0, 0], [0, pad]], constant_values="<PAD>")
  return tf.reshape(tokens, [-1, sequence_length])

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  for key in features.VOCAB_FEATURE_KEYS:
    print(key)
    print(type(inputs))
    print(inputs[key])
    embeddings = Embeddings()
    embed_path = '../data/word_embeddings-embed_size_100-epochs_100.csv'
    embeddings_dict = embeddings.get_embeddings_dict(embed_path)
    # Preserve this feature as a dense float, setting nan's to the mean.
    # outputs[features.transformed_name(key)] = tft.scale_to_z_score(
        # _fill_in_missing(inputs[key]))

  # for key in features.VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    # outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
        # _fill_in_missing(inputs[key]),
        # top_k=features.VOCAB_SIZE,
        # num_oov_buckets=features.OOV_SIZE)

  #for key, num_buckets in zip(features.BUCKET_FEATURE_KEYS,
                              #features.BUCKET_FEATURE_BUCKET_COUNT):
    #outputs[features.transformed_name(key)] = tft.bucketize(
        #_fill_in_missing(inputs[key]),
        #num_buckets)

  #for key in features.CATEGORICAL_FEATURE_KEYS:
    #outputs[features.transformed_name(key)] = _fill_in_missing(inputs[key])

  # TODO(b/157064428): Support label transformation for Keras.
  # Do not apply label transformation as it will result in wrong evaluation.
  # outputs[features.transformed_name(
      #features.LABEL_KEY)] = inputs[features.LABEL_KEY]
    tokens = tokenize_issues(_fill_in_missing(inputs["issue"], ''))
    #print(tokens)
    #print(tf.strings.as_string(inputs[key]))
    outputs[key] = tft.compute_and_apply_vocabulary(
      tokens,
      top_k=VOCAB_SIZE,
      num_oov_buckets=OOV_SIZE)

  return outputs
