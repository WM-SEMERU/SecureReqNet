<<<<<<< HEAD
docker run --name=SecureReqNet -p 8501:8501 \
--mount type=bind,source=/home/roger/serving/securereqnet/gamma,target=/models/gamma \
--mount type=bind,source=/home/roger/serving/securereqnet/alpha,target=/models/alpha \
--mount type=bind,source=/home/roger/serving/securereqnet/model_config.config,target=/models/model_config.config \
-t tensorflow/serving \
--model_config_file=/models/model_config.config