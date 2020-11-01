import csv
from tensorflow.keras.preprocessing import text
from nltk.corpus import gutenberg
from string import punctuation
from tensorflow.keras.preprocessing.sequence import skipgrams
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
englishStemmer=SnowballStemmer("english")
from securereqnet import utils
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

  train_batch_size = 1
  eval_batch_size = 1

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
    
  #train_dataset = _input_fn(  # pylint: disable=g-long-lambda
  #    train_records_list,
  #    tf_transform_output,
  #    batch_size=train_batch_size)

  #eval_dataset = _input_fn(  # pylint: disable=g-long-lambda
  #    eval_records_list,
  #    tf_transform_output,
  #    batch_size=eval_batch_size)

  model_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), '../pretrained_models/alpha')
  model = tf.keras.models.load_model(model_path)
  #model.trainable = True
  #print(model.trainable_variables)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  filepath = "../08_test/best_model.hdf5"

  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
  mc = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
  callbacks_list = [es,mc]

  dataset_list = generate_dataset()

  print(fn_args.serving_model_dir)
  
  model.fit(
            x = dataset_list[0],
            y = dataset_list[1],
            batch_size=1,
            epochs=20, #5 <------ Hyperparameter
            validation_split=0.2,
            callbacks=callbacks_list
  )

  model.save(fn_args.serving_model_dir, save_format='tf')



def generate_dataset():
  #../data replaces datasets for the to access data
  path = "../data/combined_dataset/"
  process_unit = utils.Processing_Dataset(path)
  ground_truth = process_unit.get_ground_truth()

  dataset = utils.Dynamic_Dataset(ground_truth, path,False)

  #As the data is stored in a zip file isZip = True
  test, train = process_unit.get_test_and_training(ground_truth,isZip = True)

  nltk.download('stopwords')

  embeddings = utils.Embeddings()
  max_words = 5000 #<------- [Parameter]
  pre_corpora_train = [doc for doc in train if len(doc[1])< max_words]
  pre_corpora_test = [doc for doc in test if len(doc[1])< max_words]

  embed_path = '../data/word_embeddings-embed_size_100-epochs_100.csv'
  embeddings_dict = embeddings.get_embeddings_dict(embed_path)
  
  # .decode("utf-8") takes the doc's which are saved as byte files and converts them into strings for tokenization
  corpora_train = [embeddings.vectorize(doc[1].decode("utf-8"), embeddings_dict) for doc in pre_corpora_train]#vectorization Inputs
  corpora_test = [embeddings.vectorize(doc[1].decode("utf-8"), embeddings_dict) for doc in pre_corpora_test]#vectorization

  target_train = [[int(list(doc[0])[1]),int(list(doc[0])[3])] for doc in pre_corpora_train]#vectorization Output
  target_test = [[int(list(doc[0])[1]),int(list(doc[0])[3])]for doc in pre_corpora_test]#vectorization Output
  #target_train

  max_len_sentences_train = max([len(doc) for doc in corpora_train]) #<------- [Parameter]
  max_len_sentences_test = max([len(doc) for doc in corpora_test]) #<------- [Parameter]

  max_len_sentences = max(max_len_sentences_train,max_len_sentences_test)

  min_len_sentences_train = min([len(doc) for doc in corpora_train]) #<------- [Parameter]
  min_len_sentences_test = min([len(doc) for doc in corpora_test]) #<------- [Parameter]

  min_len_sentences = max(min_len_sentences_train,min_len_sentences_test)

  embed_size = np.size(corpora_train[0][0])

  #BaseLine Architecture <-------
  embeddigs_cols = 100
  input_sh = (618,100,1)
  max_len_sentences = 618
  #Selecting filters? 
  #https://stackoverflow.com/questions/48243360/how-to-determine-the-filter-parameter-in-the-keras-conv2d-function
  #https://stats.stackexchange.com/questions/196646/what-is-the-significance-of-the-number-of-convolution-filters-in-a-convolutional

  N_filters = 128 # <-------- [HyperParameter] Powers of 2 Numer of Features
  K = 2 # <-------- [HyperParameter] Number of Classess
  
  #Data set organization
  from tempfile import mkdtemp
  import os.path as path

  #Memoization 
  file_corpora_train_x = path.join(mkdtemp(), 'alex-res-adapted-003_temp_corpora_train_x.dat') #Update per experiment
  file_corpora_test_x = path.join(mkdtemp(), 'alex-res-adapted-003_temp_corpora_test_x.dat')

  #Shaping
  shape_train_x = (len(corpora_train),max_len_sentences,embeddigs_cols,1)
  shape_test_x = (len(corpora_test),max_len_sentences,embeddigs_cols,1)

  #Data sets
  corpora_train_x = np.memmap(
        filename = file_corpora_train_x, 
        dtype='float32', 
        mode='w+', 
        shape = shape_train_x)

  corpora_test_x = np.memmap( #Test Corpora (for future evaluation)
        filename = file_corpora_test_x, 
        dtype='float32', 
        mode='w+', 
        shape = shape_test_x)

  target_train_y = np.array(target_train) #Train Target
  target_test_y = np.array(target_test) #Test Target (for future evaluation)

  #Reshaping Train Inputs
  for doc in range(len(corpora_train)):
    #print(corpora_train[doc].shape[1])
    for words_rows in range(corpora_train[doc].shape[0]):
        embed_flatten = np.array(corpora_train[doc][words_rows]).flatten() #<--- Capture doc and word
        for embedding_cols in range(embed_flatten.shape[0]):
            corpora_train_x[doc,words_rows,embedding_cols,0] = embed_flatten[embedding_cols]

  #Reshaping Test Inputs (for future evaluation)
  for doc in range(len(corpora_test)):
    for words_rows in range(corpora_test[doc].shape[0]):
        embed_flatten = np.array(corpora_test[doc][words_rows]).flatten() #<--- Capture doc and word
        for embedding_cols in range(embed_flatten.shape[0]):
            corpora_test_x[doc,words_rows,embedding_cols,0] = embed_flatten[embedding_cols]

  return [corpora_train_x, target_train_y]