import flask
import requests
import json
from flask import request, jsonify
from tempfile import mkdtemp
import os.path as path
import tensorflow as tf
from securereqnet.utils import Embeddings
import base64
import nltk
nltk.download('stopwords')
import numpy as np
from waitress import serve

app = flask.Flask(__name__)

# default route, we probably will get rid of this
@app.route('/', methods=['GET'])
def home():
    return '<h1>SecureReqNet</h1><p>Flask backend</p>'

# alpha model
@app.route('/models/alpha', methods=['POST'])
def alpha():
    content = request.get_json()
    #print(content['instances'])
    sentences = content['instances']
    
    #processed = []
    #for sentence in sentences:4
    #    processed.append(preprocess(sentence))
    
    payload = {
        "instances": preprocess(sentences[0]).tolist()
    }
    #print(payload)
    r = requests.post('http://localhost:8501/v1/models/alpha:predict', json = payload)
    model_preds = json.loads(r.content.decode('utf-8'))
    print(model_preds)
    
    preds = []

    # decode predictions
    for pred in model_preds['predictions']:
        preds.append(decode(pred))

    output = {
        "predictions": preds
    }

    return output

# gamma model
@app.route('/models/gamma', methods=['POST'])
def gamma():
    content = request.get_json()
    print(content['instances'])
    example = serialize_text(content['instances'][0])
    payload = {
      "signature_name":"serving_default",
      "instances":[
        {
          "examples":{"b64": base64.b64encode(example).decode('utf-8')}
        }
      ]
    }
    print(payload)
    r = requests.post('http://localhost:8501/v1/models/gamma:predict', json = payload)
    model_preds = json.loads(r.content.decode('utf-8'))

    
    preds = []

    # decode predictions
    for entry in model_preds['predictions']:
        preds.append([(entry["predictions"][0]>=.5),"Probability: " + str(entry["probabilities"][0])])

    output = {
        "predictions": preds
    }

    return output

## may be changed later
def decode(input):
    return float(input[0])>float(input[1])

def preprocess(sentence):
    return preprocess_placeholder(sentence)

## WILL BE REPLACED IN FUTURE BY IMPORTING TFX TRANSFORM FN
def preprocess_placeholder(sentence):
    embeddings = Embeddings()
    embed_path = 'word_embeddings-embed_size_100-epochs_100.csv'
    embeddings_dict = embeddings.get_embeddings_dict(embed_path)
    vectorized = embeddings.vectorize(sentence, embeddings_dict)

    inp_shape = (1, 618, 100, 1)

    inp = np.zeros(shape=inp_shape, dtype='float32')
    for words_rows in range(vectorized.shape[0]):
        embed_flatten = np.array(vectorized[words_rows]).flatten()
        for embedding_cols in range(embed_flatten.shape[0]):
            inp[0,words_rows,embedding_cols,0] = embed_flatten[embedding_cols]
    # print(inp)
    return inp

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_text(text):
    if type(text) == str:
        text = bytes(text, 'utf-8')
    example = tf.train.Example(features=tf.train.Features(feature={
        'text': _bytes_feature(text)
        }))
    serialized_example = example.SerializeToString()
    return serialized_example

if __name__ == "__main__":
    # app.run()
    serve(app, host = '0.0.0.0', port = 8055)