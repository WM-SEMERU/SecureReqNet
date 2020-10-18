import tensorflow_transform as tft
import tensorflow as tf
from tfx.components.trainer import executor
import os

# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
  files = os.listdir("C:\\Users\\John\\Documents\\GitHub\\SecureReqNet\\data\\records\\tfrecords_train")

  #for element in dataset:
  #  print(element)
  """Build the estimator using the high level API.

  Args:
    trainer_fn_args: Holds args used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.

  Returns:
    A dict of the following:

      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 32
  num_dnn_layers = 4
  dnn_decay_factor = 0.7

  train_batch_size = 40
  eval_batch_size = 40

  tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

  # Provides input data for training as minibatches. 
  train_records_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '..\\data\\records\\tfrecords_train')
  train_records_list = os.listdir(train_records_path)
  for i in range(0,len(train_records_list)):
    train_records_list[i] = os.path.join(train_records_path, train_records_list[i])

  eval_records_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '..\\data\\records\\tfrecords_eval')
  eval_records_list = os.listdir(eval_records_path)
  for i in range(0,len(eval_records_list)):
    eval_records_list[i] = os.path.join(eval_records_path, eval_records_list[i])

  test = tf.data.TFRecordDataset(train_records_list)
  for raw_record in test.take(10):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

  train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      train_records_list,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      eval_records_list,
      tf_transform_output,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
      train_input_fn,
      max_steps=trainer_fn_args.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('securereqnet', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=trainer_fn_args.eval_steps,
      exporters=[exporter],
      name='securereqnet-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)
  #warm_start_from = trainer_fn_args.base_models[
  #    0] if trainer_fn_args.base_models else None

  model_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '../pretrained_models/alpha.hdf5')

  estimator = tf.keras.estimator.model_to_estimator(
    keras_model_path=model_path, custom_objects=None, model_dir=None,
    config=None, checkpoint_format='checkpoint', metric_names_map=None
  )


  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }

# Provides input data for training as minibatches.
def _input_fn(training_files, transform_output, batch_size):
  print("")

