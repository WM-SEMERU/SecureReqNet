import tensorflow as tf
import sys

#input the path to the tfrecord you want to examine
raw_dataset = tf.data.TFRecordDataset(sys.argv[1])
for raw_record in raw_dataset.take(10):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)