docker run -p 8502:8501 
--mount type=bind, source=/home/semeru/securereqnet/serving/saved_model_alpha, target=/models/alpha,\
--mount type=bind, source=/home/semeru/securereqnet/serving/saved_model_gamma, target=/models/gamma,\
--mount type=bind, source=/home/semeru/securereqnet/serving/model_config.config, target=/models/model_config.config,\
-t tensorflow/serving &,\
--model_config_file=/models/model_config.config