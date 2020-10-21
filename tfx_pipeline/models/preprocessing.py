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
import numpy
import pandas
import re
from string import punctuation
import nltk
from nltk.stem.snowball import SnowballStemmer

SEQUENCE_LENGTH = 100
VOCAB_SIZE = 10000
OOV_SIZE = 100

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
    
  #print("SHAPE:")
  #print(inputs["x"].shape)
  #x = tf.reshape(inputs['x'],(1,618,100,1))
  #print("SHAPED")
    
  #print("x's shape is:")
  #print(x.shape)
  #outputs['x'] = x
  outputs['x'] = inputs["x"]
  outputs["numberOfSamples"] = inputs["numberOfSamples"]
  outputs['y'] = inputs['y']
    

  return outputs
