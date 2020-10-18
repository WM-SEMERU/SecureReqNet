import tensorflow_transform as tft
import tensorflow as tf
import numpy as np
from tfx.components.trainer import executor
import os

# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
  files = os.listdir("C:\\Users\\John\\Documents\\GitHub\\SecureReqNet\\data\\records\\tfrecords_train")

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
  #first_dnn_layer_size = 32
  #num_dnn_layers = 4
  #dnn_decay_factor = 0.7

  train_batch_size = 40
  eval_batch_size = 40

  tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

  # Collects a list of the paths to tfrecord files in the eval directory in train_records_list
  train_records_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '..\\data\\records\\tfrecords_train')
  train_records_list = os.listdir(train_records_path)
  for i in range(0,len(train_records_list)):
    train_records_list[i] = os.path.join(train_records_path, train_records_list[i])

  # Collects a list of the paths to tfrecord files in the eval directory in eval_records_list
  eval_records_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '..\\data\\records\\tfrecords_eval')
  eval_records_list = os.listdir(eval_records_path)
  for i in range(0,len(eval_records_list)):
    eval_records_list[i] = os.path.join(eval_records_path, eval_records_list[i])
  
  train_eval_records = [train_records_list, eval_records_list]
  # This iterates through both the list of train and eval tfrecord paths, and uses the resulting 
  # x and y tensors from each record to make combined tensors for x and y where the first dimension
  # is split by record, as per the dataset requirements. 
  loop_num = 0
  for record_list in train_eval_records:
    record_x = []
    record_y = []

    records = tf.data.TFRecordDataset(record_list)
    for raw_record in records.take(10):
      example = tf.train.Example()
      example.ParseFromString(raw_record.numpy())
      record_values = parse_tfrecord_string(str(example))
      record_x.append(record_values[0])
      record_y.append(record_values[1])
      print(len(record_y))

    full_x = record_x[0]
    for i in range(1, len(record_x)-1):
      full_x = tf.concat([full_x, record_x[i]], 0)

    # If there's only one record, len(record_x)-1 doesn't work. Temporary
    if(loop_num == 0):
      full_x = tf.reshape(full_x, [len(record_x)-1, 618,100,1])
    else:
      full_x = tf.reshape(full_x, [1, 618,100,1])

    print(len(record_y))
    full_y = record_y[0]
    for i in range(1, len(record_y)-1):
      full_y = tf.stack(full_y)
      print(full_y)
    #full_y = tf.reshape(full_y, [len(record_y), ])

    print(full_x.get_shape())
    print(full_y.get_shape())
    dataset = (full_x, full_y)
    if(loop_num == 0):
      train_dataset = dataset
    else:
      eval_dataset = dataset
    loop_num += 1

  train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      train_dataset,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      eval_dataset,
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

# Parses the string output of a tfrecord and returns the x values and y values in a list
def parse_tfrecord_string(record_string):
  #print(record_string.splitlines(False))
  lines = record_string.splitlines(False)
  x_index = 0
  y_index = 0
  y_end_index = 0
  scanningX = False
  scanningY = False
  for i in range(0, len(lines)):
    if('key: "x"' in lines[i]):
      scanningX = True
      scanningY = False
      x_index = i+3
    elif('key: "y"' in lines[i]):
      scanningX = False
      scanningY = True
      y_index = i+3
    # Since the length of the y values is variable, we need to record the index of
    # the final value so we know how much to slice.
    if(scanningY and "}" in lines[i]):
      y_end_index = i
      scanningY = False
    if("value: " in lines[i] and scanningX):
      lines[i] =  float(lines[i].replace("value: ", "").strip())
    elif("value: " in lines[i]):
      lines[i] =  int(lines[i].replace("value: ", "").strip())

  x_slice = lines[x_index:x_index+61800]
  y_slice = lines[y_index:y_end_index]

  x_tensor = tf.convert_to_tensor(x_slice, dtype=tf.float32)
  y_tensor = tf.convert_to_tensor(y_slice)

  return [x_tensor, y_tensor]
