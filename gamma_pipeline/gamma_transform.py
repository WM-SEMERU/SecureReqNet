
import tensorflow as tf

import tensorflow_transform as tft

SEQUENCE_LENGTH = 100
VOCAB_SIZE = 10000
OOV_SIZE = 100

def process_sentence(sentences):
    sentences = tf.strings.lower(sentences)
    sentences = tf.strings.regex_replace(sentences, r" '| '|^'|'$", " ")
    sentences = tf.strings.regex_replace(sentences,"\n","")
    nums = '0123456789'
    for item in nums:
        sentences = tf.strings.regex_replace(sentences, item, "")
    for item in '"#$%&,-/:;<=>@_`|~:'+ "'":
        sentences = tf.strings.regex_replace(sentences,item,"")
    for item in "()*+?[][]{}^\!":
        sentences = tf.strings.regex_replace(sentences,"\\" + item,"")
    sentences = tf.strings.regex_replace(sentences,"\\.","")
    
    sentences = tf.strings.strip(sentences)
    return sentences
    
def tokenize_text(text, sequence_length=SEQUENCE_LENGTH):
  text = tf.strings.lower(text)
  text = tf.strings.regex_replace(text, r" '| '|^'|'$", " ")
  text = tf.strings.regex_replace(text, "[^a-z' ]", " ")
  text = process_sentence(text)
  tokens = tf.strings.split(text)[:, :sequence_length]
  start_tokens = tf.fill([tf.shape(text)[0], 1], "<START>")
  end_tokens = tf.fill([tf.shape(text)[0], 1], "<END>")
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
  outputs["id"] = inputs["id"]
  tokens = tokenize_text(_fill_in_missing(inputs["text"], ''))
  outputs["text_xf"] = tft.compute_and_apply_vocabulary(
      tokens,
      top_k=VOCAB_SIZE,
      num_oov_buckets=OOV_SIZE)
  outputs["label_xf"] = _fill_in_missing(inputs["label"], -1)
  return outputs

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
