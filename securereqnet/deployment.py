# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10_deployment.ipynb (unless otherwise specified).

__all__ = ['__get_predictions', '__decode', 'create_app', 'serve']

# Cell
import flask
from flask import Flask
import requests
import json
from flask import request, jsonify
from tempfile import mkdtemp
import os.path as path
from .preprocessing import vectorize_sentences
import numpy as np
from waitress import serve

# Cell

def __get_predictions(sentences, endpoint_uri):
    payload = {
            "instances": vectorize_sentences(sentences).tolist()
        }

    print(payload)
    r = requests.post(endpoint_uri, json = payload)
    model_preds = json.loads(r.content.decode('utf-8'))


    preds = []

    # decode predictions
    for pred in model_preds['predictions']:
        preds.append(__decode(pred))

    output = {
        "predictions": preds
    }

    return output




# Cell

def __decode(input):
    return float(input[0])>float(input[1])

# Cell

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        TFX_ENDPOINT='http://localhost:8503/v1/models/alpha:predict'
    )

    if test_config:
        app.config.from_mapping(test_config)
    else:
        app.config.from_pyfile('config.py', silent=True)

    # default route, we probably will get rid of this
    @app.route('/', methods=['GET'])
    def home():
        return '<h1>SecureReqNet</h1><p>Flask backend</p>'

    # alpha model
    @app.route('/models/alpha', methods=['POST'])
    def alpha():
        content = request.get_json()
        print(content['instances'])
        sentences = content['instances']
        return __get_predictions(sentences, app.config['TFX_ENDPOINT'])

    return app




# Cell

def serve(host, port):
    serve(create_app(), host = host, port = port)