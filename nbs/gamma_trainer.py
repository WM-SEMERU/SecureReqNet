
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
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import gamma_constants

key = "sentence"
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


def _build_keras_model(train_ds):
  """Creates a DNN Keras model for classifying taxi data.

  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).

  Returns:
    A keras Model.
  """
  vocab_size = 18000
  sequence_length = 618

    # Use the text vectorization layer to normalize, split, and map strings to 
    # integers. Note that the layer uses the custom standardization defined above. 
    # Set maximum_sequence length as all samples are not of the same length.
  #vectorize_layer = TextVectorization(
        #max_tokens=vocab_size,
        #output_mode='int',
        #output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
  #text_ds = train_ds.map(lambda x, y: x)
  #vectorize_layer.adapt(text_ds)

  model = Sequential()
  model.add(tf.keras.layers.Embedding(190010, 40, input_length=618,mask_zero=True))
  model.add(layers.GlobalAveragePooling1D())

  model.add(Dense(16, activation='relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(1, activation='softmax'))
  model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
  model.summary(print_fn=absl.logging.info)
    
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """


  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  print("\n\nTHISHERE")
  one_hot_columns = [
      tf.feature_column.categorical_column_with_vocabulary_file(
          key=key,
          vocabulary_file=tf_transform_output.vocabulary_file_by_name(
              vocab_filename="sentence_key"))]
  print(one_hot_columns)
    
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, 
                            tf_transform_output, 40)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, 
                           tf_transform_output, 40)

  model = _build_keras_model(train_dataset)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      shuffle=True,
      workers=10,
      callbacks=[tensorboard_callback], verbose=0)

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

