
from typing import List, Text

import os
import absl
import datetime
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from keras.models import Sequential
from keras.layers import Dense
from tfx_bsl.tfxio import dataset_options

import gamma_constants

_VOCAB_FEATURE_KEYS = gamma_constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = gamma_constants.VOCAB_SIZE
_OOV_SIZE = gamma_constants.OOV_SIZE
_LABEL_KEY = gamma_constants.LABEL_KEY
_transformed_name = gamma_constants.transformed_name


def _transformed_names(keys):
  return [_transformed_name(key) for key in keys]


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  model.tft_layer = tf_transform_output.transform_features_layer()
  print("TFT LAYER SHAPE: " + model.tft_layer.shape)

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)

  return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema)


def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying taxi data.

  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).

  Returns:
    A keras Model.
  """

  categorical_columns = [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)
      for key in _transformed_names(_VOCAB_FEATURE_KEYS)
  ]
  indicator_column = [
      tf.feature_column.indicator_column(categorical_column)
      for categorical_column in categorical_columns
  ]

  # TODO(b/139668410) replace with premade wide_and_deep keras model
  wide_columns=indicator_column,
  dnn_hidden_units=hidden_units or [100, 70, 50, 25]
    
  input_layers = {
      colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
      for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
  }

  output = tf.keras.layers.Dense(
      1, activation='softmax')

  model = Sequential()
  #model.add(Deinput_layers)
  model.add(Dense(12, input_dim=categorical_columns.shape, activation='relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(lr=0.001),
      metrics=[tf.keras.metrics.BinaryAccuracy()])
  model.summary(print_fn=absl.logging.info)
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 100
  num_dnn_layers = 4
  dnn_decay_factor = 0.7

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, 
                            tf_transform_output, 40)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, 
                           tf_transform_output, 40)

  model = _build_keras_model(
      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ])

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
