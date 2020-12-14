
import neural_structured_learning as nsl

import tensorflow as tf

import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils


NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'
LABEL_KEY = 'label'
ID_FEATURE_KEY = 'id'

def _transformed_name(key):
  return key + '_xf'


def _transformed_names(keys):
  return [_transformed_name(key) for key in keys]


# Hyperparameters:
#
# We will use an instance of `HParams` to inclue various hyperparameters and
# constants used for training and evaluation. We briefly describe each of them
# below:
#
# -   max_seq_length: This is the maximum number of words considered from each
#                     movie review in this example.
# -   vocab_size: This is the size of the vocabulary considered for this
#                 example.
# -   oov_size: This is the out-of-vocabulary size considered for this example.
# -   distance_type: This is the distance metric used to regularize the sample
#                    with its neighbors.
# -   graph_regularization_multiplier: This controls the relative weight of the
#                                      graph regularization term in the overall
#                                      loss function.
# -   num_neighbors: The number of neighbors used for graph regularization. This
#                    value has to be less than or equal to the `num_neighbors`
#                    argument used above in the GraphAugmentation component when
#                    invoking `nsl.tools.pack_nbrs`.
# -   num_fc_units: The number of units in the fully connected layer of the
#                   neural network.
class HParams(object):
  """Hyperparameters used for training."""
  def __init__(self):
    ### dataset parameters
    # The following 3 values should match those defined in the Transform
    # Component.
    self.max_seq_length = 100
    self.vocab_size = 10000
    self.oov_size = 100
    ### Neural Graph Learning parameters
    self.distance_type = nsl.configs.DistanceType.L2
    self.graph_regularization_multiplier = 0.1
    # The following value has to be at most the value of 'num_neighbors' used
    # in the GraphAugmentation component.
    self.num_neighbors = 1
    ### Model Architecture
    self.num_embedding_dims = 16
    self.num_fc_units = 64

HPARAMS = HParams()


def optimizer_fn():
  """Returns an instance of `tf.Optimizer`."""
  return tf.compat.v1.train.RMSPropOptimizer(
    learning_rate=0.0001, decay=1e-6)


def build_train_op(loss, global_step):
  """Builds a train op to optimize the given loss using gradient descent."""
  with tf.name_scope('train'):
    optimizer = optimizer_fn()
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
  return train_op


# Building the model:
#
# A neural network is created by stacking layers—this requires two main
# architectural decisions:
# * How many layers to use in the model?
# * How many *hidden units* to use for each layer?
#
# In this example, the input data consists of an array of word-indices. The
# labels to predict are either 0 or 1. We will use a feed-forward neural network
# as our base model in this tutorial.
def feed_forward_model(features, is_training, reuse=tf.compat.v1.AUTO_REUSE):
  """Builds a simple 2 layer feed forward neural network.

  The layers are effectively stacked sequentially to build the classifier. The
  first layer is an Embedding layer, which takes the integer-encoded vocabulary
  and looks up the embedding vector for each word-index. These vectors are
  learned as the model trains. The vectors add a dimension to the output array.
  The resulting dimensions are: (batch, sequence, embedding). Next is a global
  average pooling 1D layer, which reduces the dimensionality of its inputs from
  3D to 2D. This fixed-length output vector is piped through a fully-connected
  (Dense) layer with 16 hidden units. The last layer is densely connected with a
  single output node. Using the sigmoid activation function, this value is a
  float between 0 and 1, representing a probability, or confidence level.

  Args:
    features: A dictionary containing batch features returned from the
      `input_fn`, that include sample features, corresponding neighbor features,
      and neighbor weights.
    is_training: a Python Boolean value or a Boolean scalar Tensor, indicating
      whether to apply dropout.
    reuse: a Python Boolean value for reusing variable scope.

  Returns:
    logits: Tensor of shape [batch_size, 1].
    representations: Tensor of shape [batch_size, _] for graph regularization.
      This is the representation of each example at the graph regularization
      layer.
  """

  with tf.compat.v1.variable_scope('ff', reuse=reuse):
    inputs = features[_transformed_name('text')]
    embeddings = tf.compat.v1.get_variable(
        'embeddings',
        shape=[
            HPARAMS.vocab_size + HPARAMS.oov_size, HPARAMS.num_embedding_dims
        ])
    embedding_layer = tf.nn.embedding_lookup(embeddings, inputs)

    pooling_layer = tf.compat.v1.layers.AveragePooling1D(
        pool_size=HPARAMS.max_seq_length, strides=HPARAMS.max_seq_length)(
            embedding_layer)
    # Shape of pooling_layer is now [batch_size, 1, HPARAMS.num_embedding_dims]
    pooling_layer = tf.reshape(pooling_layer, [-1, HPARAMS.num_embedding_dims])

    dense_layer = tf.compat.v1.layers.Dense(
        16, activation='relu')(
            pooling_layer)

    output_layer = tf.compat.v1.layers.Dense(
        1, activation='sigmoid')(
            dense_layer)

    # Graph regularization will be done on the penultimate (dense) layer
    # because the output layer is a single floating point number.
    return output_layer, dense_layer


# A note on hidden units:
#
# The above model has two intermediate or "hidden" layers, between the input and
# output, and excluding the Embedding layer. The number of outputs (units,
# nodes, or neurons) is the dimension of the representational space for the
# layer. In other words, the amount of freedom the network is allowed when
# learning an internal representation. If a model has more hidden units
# (a higher-dimensional representation space), and/or more layers, then the
# network can learn more complex representations. However, it makes the network
# more computationally expensive and may lead to learning unwanted
# patterns—patterns that improve performance on training data but not on the
# test data. This is called overfitting.


# This function will be used to generate the embeddings for samples and their
# corresponding neighbors, which will then be used for graph regularization.
def embedding_fn(features, mode):
  """Returns the embedding corresponding to the given features.

  Args:
    features: A dictionary containing batch features returned from the
      `input_fn`, that include sample features, corresponding neighbor features,
      and neighbor weights.
    mode: Specifies if this is training, evaluation, or prediction. See
      tf.estimator.ModeKeys.

  Returns:
    The embedding that will be used for graph regularization.
  """
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  _, embedding = feed_forward_model(features, is_training)
  return embedding


def feed_forward_model_fn(features, labels, mode, params, config):
  """Implementation of the model_fn for the base feed-forward model.

  Args:
    features: This is the first item returned from the `input_fn` passed to
      `train`, `evaluate`, and `predict`. This should be a single `Tensor` or
      `dict` of same.
    labels: This is the second item returned from the `input_fn` passed to
      `train`, `evaluate`, and `predict`. This should be a single `Tensor` or
      `dict` of same (for multi-head models). If mode is `ModeKeys.PREDICT`,
      `labels=None` will be passed. If the `model_fn`'s signature does not
      accept `mode`, the `model_fn` must still be able to handle `labels=None`.
    mode: Optional. Specifies if this training, evaluation or prediction. See
      `ModeKeys`.
    params: An HParams instance as returned by get_hyper_parameters().
    config: Optional configuration object. Will receive what is passed to
      Estimator in `config` parameter, or the default `config`. Allows updating
      things in your model_fn based on configuration such as `num_ps_replicas`,
      or `model_dir`. Unused currently.

  Returns:
     A `tf.estimator.EstimatorSpec` for the base feed-forward model. This does
     not include graph-based regularization.
  """

  is_training = mode == tf.estimator.ModeKeys.TRAIN

  # Build the computation graph.
  probabilities, _ = feed_forward_model(features, is_training)
  predictions = tf.round(probabilities)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # labels will be None, and no loss to compute.
    cross_entropy_loss = None
    eval_metric_ops = None
  else:
    # Loss is required in train and eval modes.
    # Flatten 'probabilities' to 1-D.
    probabilities = tf.reshape(probabilities, shape=[-1])
    cross_entropy_loss = tf.compat.v1.keras.losses.binary_crossentropy(
        labels, probabilities)
    eval_metric_ops = {
        'accuracy': tf.compat.v1.metrics.accuracy(labels, predictions)
    }

  if is_training:
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_op = build_train_op(cross_entropy_loss, global_step)
  else:
    train_op = None

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions={
          'probabilities': probabilities,
          'predictions': predictions
      },
      loss=cross_entropy_loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')


def _example_serving_receiver_fn(tf_transform_output, schema):
  """Build the serving in inputs.

  Args:
    tf_transform_output: A TFTransformOutput.
    schema: the schema of the input data.

  Returns:
    Tensorflow graph which parses examples, applying tf-transform to them.
  """
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_feature_spec.pop(LABEL_KEY)

  # We don't need the ID feature for serving.
  raw_feature_spec.pop(ID_FEATURE_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  transformed_features = tf_transform_output.transform_raw_features(
      serving_input_receiver.features)

  # Even though, LABEL_KEY was removed from 'raw_feature_spec', the transform
  # operation would have injected the transformed LABEL_KEY feature with a
  # default value.
  transformed_features.pop(_transformed_name(LABEL_KEY))
  return tf.estimator.export.ServingInputReceiver(
      transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(tf_transform_output, schema):
  """Build everything needed for the tf-model-analysis to run the model.

  Args:
    tf_transform_output: A TFTransformOutput.
    schema: the schema of the input data.

  Returns:
    EvalInputReceiver function, which contains:
      - Tensorflow graph which parses raw untransformed features, applies the
        tf-transform preprocessing operators.
      - Set of raw, untransformed features.
      - Label against which predictions will be compared.
  """
  # Notice that the inputs are raw features, not transformed features here.
  raw_feature_spec = _get_raw_feature_spec(schema)

  # We don't need the ID feature for TFMA.
  raw_feature_spec.pop(ID_FEATURE_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  transformed_features = tf_transform_output.transform_raw_features(
      serving_input_receiver.features)

  labels = transformed_features.pop(_transformed_name(LABEL_KEY))
  return tfma.export.EvalInputReceiver(
      features=transformed_features,
      receiver_tensors=serving_input_receiver.receiver_tensors,
      labels=labels)


def _augment_feature_spec(feature_spec, num_neighbors):
  """Augments `feature_spec` to include neighbor features.
    Args:
      feature_spec: Dictionary of feature keys mapping to TF feature types.
      num_neighbors: Number of neighbors to use for feature key augmentation.
    Returns:
      An augmented `feature_spec` that includes neighbor feature keys.
  """
  for i in range(num_neighbors):
    feature_spec['{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'id')] = \
        tf.io.VarLenFeature(dtype=tf.string)
    # We don't care about the neighbor features corresponding to
    # _transformed_name(LABEL_KEY) because the LABEL_KEY feature will be
    # removed from the feature spec during training/evaluation.
    feature_spec['{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'text_xf')] = \
        tf.io.FixedLenFeature(shape=[HPARAMS.max_seq_length], dtype=tf.int64,
                              default_value=tf.constant(0, dtype=tf.int64,
                                                        shape=[HPARAMS.max_seq_length]))
    # The 'NL_num_nbrs' features is currently not used.

  # Set the neighbor weight feature keys.
  for i in range(num_neighbors):
    feature_spec['{}{}{}'.format(NBR_FEATURE_PREFIX, i, NBR_WEIGHT_SUFFIX)] = \
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=[0.0])

  return feature_spec


def _input_fn(filenames, tf_transform_output, is_training, batch_size=200):
  """Generates features and labels for training or evaluation.

  Args:
    filenames: [str] list of CSV files to read data from.
    tf_transform_output: A TFTransformOutput.
    is_training: Boolean indicating if we are in training mode.
    batch_size: int First dimension size of the Tensors returned by input_fn

  Returns:
    A (features, indices) tuple where features is a dictionary of
      Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  # During training, NSL uses augmented training data (which includes features
  # from graph neighbors). So, update the feature spec accordingly. This needs
  # to be done because we are using different schemas for NSL training and eval,
  # but the Trainer Component only accepts a single schema.
  if is_training:
    transformed_feature_spec =_augment_feature_spec(transformed_feature_spec,
                                                    HPARAMS.num_neighbors)

  dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

  transformed_features = tf.compat.v1.data.make_one_shot_iterator(
      dataset).get_next()
  # We pop the label because we do not want to use it as a feature while we're
  # training.
  return transformed_features, transformed_features.pop(
      _transformed_name(LABEL_KEY))


# TFX will call this function
def trainer_fn(hparams, schema):
  """Build the estimator using the high level API.
  Args:
    hparams: Holds hyperparameters used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.
  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  train_batch_size = 40
  eval_batch_size = 40

  tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

  train_input_fn = lambda: _input_fn(
      hparams.train_files,
      tf_transform_output,
      is_training=True,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(
      hparams.eval_files,
      tf_transform_output,
      is_training=False,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn,
      max_steps=hparams.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(
      tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('gamma', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=hparams.eval_steps,
      exporters=[exporter],
      name='gamma-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=hparams.serving_model_dir)

  estimator = tf.estimator.Estimator(
      model_fn=feed_forward_model_fn, config=run_config, params=HPARAMS)

  # Create a graph regularization config.
  graph_reg_config = nsl.configs.make_graph_reg_config(
      max_neighbors=HPARAMS.num_neighbors,
      multiplier=HPARAMS.graph_regularization_multiplier,
      distance_type=HPARAMS.distance_type,
      sum_over_axis=-1)

  # Invoke the Graph Regularization Estimator wrapper to incorporate
  # graph-based regularization for training.
  graph_nsl_estimator = nsl.estimator.add_graph_regularization(
      estimator,
      embedding_fn,
      optimizer_fn=optimizer_fn,
      graph_reg_config=graph_reg_config)

  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(
      tf_transform_output, schema)

  return {
      'estimator': graph_nsl_estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }
