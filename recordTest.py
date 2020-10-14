import tensorflow as tf
raw_dataset = tf.data.TFRecordDataset("nbs/testdata.tfrecord")
for raw_record in raw_dataset.take(10):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)