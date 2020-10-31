import tensorflow_transform as tft
import tensorflow as tf
import numpy as np
from tfx.components.trainer import executor
import os
from tfx.components.trainer.executor import TrainerFnArgs
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dot, Input, Dense, Reshape, LSTM, Conv2D, Flatten, MaxPooling1D, Dropout, MaxPooling2D
from tensorflow.keras.layers import Embedding, Multiply, Subtract
from tensorflow.keras.models import Sequential, Model

# TFX will call this function
def run_fn(fn_args: TrainerFnArgs):
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_batch_size = 40
  eval_batch_size = 40

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
    
  train_dataset = _input_fn(  # pylint: disable=g-long-lambda
      train_records_list,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_dataset = _input_fn(  # pylint: disable=g-long-lambda
      eval_records_list,
      tf_transform_output,
      batch_size=eval_batch_size)

  model_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '../pretrained_models/alpha')
  model = tf.keras.models.load_model(model_path)
  #model.trainable = True
  #print(model.trainable_variables)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  filepath = "../08_test/best_model.hdf5"

  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
  mc = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
  callbacks_list = [es,mc]

  model.fit(
            train_dataset,
            #batch_size=64,
            epochs=2000, #5 <------ Hyperparameter
            validation_data=eval_dataset,
            callbacks=callbacks_list
  )

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
  #warm_start_from = trainer_fn_args.base_models[
  #    0] if trainer_fn_args.base_models else None

  #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# This iterates through both the list of train or eval tfrecord paths, and uses the resulting 
# x and y tensors from each record to make combined tensors for x and y where the first dimension
# is split by record, as per the dataset requirements. 
def _input_fn(record_list, transform_output, batch_size):
  record_x = []
  record_y = []

  records = tf.data.TFRecordDataset(record_list)
  for raw_record in records.take(10):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    record_values = parse_tfrecord_string(str(example))
    record_x.append(record_values[0])
    record_y.append(record_values[1])

  full_x = record_x[0]
  for i in range(1, len(record_x)):
    full_x = tf.concat([full_x, record_x[i]], 0)

  full_x = tf.reshape(full_x, [len(record_x), 618,100,1])

  full_y = record_y[0]
  for i in range(1, len(record_y)-1):
    full_y = tf.stack(record_y)
    #full_y = tf.reshape(full_y, [len(record_y), ])
  print(full_x.get_shape())
  print(full_y.get_shape())
  dataset = (full_x, full_y)
  
  return dataset


# Parses the string output of a tfrecord and returns the x values and y values in a list
def parse_tfrecord_string(record_string):
  #print(record_string.splitlines(False))
  lines = record_string.splitlines(False)
  print(lines[2], lines[5])
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
    #if(scanningY):
    #  print(lines[i])
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
  y_tensor = tf.reshape(y_tensor, [1,len(y_tensor)])

  return [x_tensor, y_tensor]
