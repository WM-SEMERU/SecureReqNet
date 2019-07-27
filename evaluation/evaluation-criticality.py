#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load model
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dot, Input, Dense, Reshape, LSTM
from tensorflow.keras.layers import Embedding, Multiply, Subtract
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Lambda

from string import punctuation
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import skipgrams
import itertools

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt


# In[18]:


from sklearn.metrics import average_precision_score,precision_recall_curve
from sklearn.utils.fixes import signature


# In[25]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[2]:


path = 'results[10000]/embeds20-improvement-99-0.51.hdf5'
criticality_network = load_model(path) #<----- The Model


# In[9]:


df_history_training = pd.read_csv('history_training.csv')
np_target_test_y = np.load('target_test_y.npy')
np_corpora_test_x = np.load('corpora_test_x.npy')


# In[4]:


df_history_training.head()


# In[27]:


for elem in np_target_test_y:
    print(elem[0])


# In[11]:


np_corpora_test_x


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


####Ploting Validation (overfitting assessment)
# summarize history for accuracy
plt.plot(df_history_training['accuracy'])
plt.plot(df_history_training['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[8]:


plt.plot(df_history_training['loss'])
plt.plot(df_history_training['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[12]:


#Making Evaluations
score = criticality_network.evaluate(np_corpora_test_x, np_target_test_y, verbose=1)


# In[14]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[15]:


#Making Predictions
history_predict = criticality_network.predict(x=np_corpora_test_x)
history_predict


# In[16]:


inferred_data = pd.DataFrame(history_predict,columns=list('AB'))
target_data = pd.DataFrame(np_target_test_y,columns=list('LN'))
data = target_data.join(inferred_data)


# In[17]:


data.head()


# In[20]:


y_true = list(data['L'])
y_score= list(data['A'])
average_precision = average_precision_score(y_true, y_score)


# In[21]:


print('Average precision-recall score: {0:0.2f}'.format(average_precision))


# In[22]:


precision, recall, thresholds = precision_recall_curve(y_true, y_score)


# In[24]:


# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))


# In[26]:


#ROC Curve (all our samples are balanced)
auc = roc_auc_score(y_true, y_score)
print('AUC: %.3f' % auc)


# In[ ]:




