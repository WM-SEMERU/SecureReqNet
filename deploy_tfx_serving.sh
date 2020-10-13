docker run -p 8502:8501 --mount type=bind,\
source=/home/semeru/securereqnet/serving/saved_model_alpha,\
target=/models/alpha -e MODEL_NAME=alpha -t tensorflow/serving &