# Lint as: python2, python3
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
"""
This file defines the TFX pipeline and various components in the pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional, Text
import os

import tensorflow_model_analysis as tfma
#from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components import InfraValidator
from tfx.components.base import executor_spec
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component  # pylint: disable=unused-import
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto import infra_validator_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
#from tfx.utils.dsl_utils import external_input
#from tfx.utils.dsl_utils import csv_input

from ml_metadata.proto import metadata_store_pb2


def create_pipeline(

    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
    # query: Text,
    module_file: Text,
    preprocessing_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    eval_accuracy_threshold: float,
    loss_threshold: float,
    auc_threshold: float,
    serving_model_dir: Text,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
) -> pipeline.Pipeline:

  components = []

  # Brings data into the pipeline or otherwise joins/converts training data.

  # Ingests pre-split data based on specified file pattern
  tf_input = example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(name='train', pattern='tfrecords_train\\*'),
                    example_gen_pb2.Input.Split(name='eval', pattern='tfrecords_eval\\*')
                ])

  '''
  # Splits input data with a 90:10 train:eval ratio
  output = example_gen_pb2.Output(
                split_config=example_gen_pb2.SplitConfig(splits=[
                    example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=9),
                    example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
                ]))
  '''

  example_gen = ImportExampleGen(input_base=data_path, input_config=tf_input)
  
  # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
  # example_gen = big_query_example_gen_component.BigQueryExampleGen(
  #     query=query)
  
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  # Input: Examples from the ExampleGen component
  # Output: Dataset statistics to be used by the SchemaGen component
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  # TODO(step 5): Uncomment here to add StatisticsGen to the pipeline.
  components.append(statistics_gen)

  # Generates schema based on statistics files.
  # Input: Statistics from the StatisticsGen component\
  # Output: A schema of the model for use in the ExampleValidator, Transform, and Trainer components.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)
  # TODO(step 5): Uncomment here to add SchemaGen to the pipeline.
  components.append(schema_gen)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(  # pylint: disable=unused-variable
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  # TODO(step 5): Uncomment here to add ExampleValidator to the pipeline.
  # components.append(example_validator)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      preprocessing_fn=preprocessing_fn)
  # TODO(step 6): Uncomment here to add Transform to the pipeline.
  # components.append(transform)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer_args = {
      'module_file' : module_file,
      'examples' : example_gen.outputs['examples'],
      #'transformed_examples': transform.outputs['transformed_examples'],
      'schema': schema_gen.outputs['schema'],
      #'transform_graph': transform.outputs['transform_graph'],
      'train_args': train_args,
      'eval_args': eval_args,
      'custom_executor_spec':
          executor_spec.ExecutorClassSpec(trainer_executor.Executor),
  }
  if ai_platform_training_args is not None:
    trainer_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(
                ai_platform_trainer_executor.GenericExecutor
            ),
        'custom_config': {
            ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                ai_platform_training_args,
        }
    })
  trainer = Trainer(**trainer_args)
  # TODO(step 6): Uncomment here to add Trainer to the pipeline.
  # components.append(trainer)

  # Get the latest blessed model for model validation.
  model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))
  # TODO(step 6): Uncomment here to add ResolverNode to the pipeline.
  # components.append(model_resolver)

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='eval')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          # binary cross-entropy loss
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='BinaryCrossentropy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          upper_bound={'value': loss_threshold}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.LOWER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ]),

          # binary accuracy
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='BinaryAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': eval_accuracy_threshold}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ]),

          # AUC
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='AUC',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': auc_threshold}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
          
      ])

  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
  # TODO(step 6): Uncomment here to add Evaluator to the pipeline.
  # components.append(evaluator)


  # Launches sandboxed server with the model
  # Validates that model can be loaded and queried
  # Input: A model from the Trainer component, examples from the ExampleGen component
  # Output: A blessed model that is sent to the Pusher component

  infra_validator = InfraValidator(
      model=trainer.outputs['model'],
      examples=example_gen.outputs['examples'],
      serving_spec=infra_validator_pb2.ServingSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServing(  # Using TF Serving.
              tags=['latest']
          ),
          local_docker=infra_validator_pb2.LocalDockerConfig(),  # Running on local docker.
      ),
      validation_spec=infra_validator_pb2.ValidationSpec(
          max_loading_time_seconds=60,
          num_tries=5,
      ),
      request_spec=infra_validator_pb2.RequestSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec(),
          num_examples=1,
      )
  )
  # components.append(infra_validator)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher_args = {
      'model':
          trainer.outputs['model'],
      'model_blessing':
          evaluator.outputs['blessing'],
      # Uncomment these when deploying InfraValidator
      #'infra_blessing':
      #    infra_validator.outputs['blessing'],
      'push_destination':
          pusher_pb2.PushDestination(
              filesystem=pusher_pb2.PushDestination.Filesystem(
                  base_directory=serving_model_dir)),
  }
  pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
  # TODO(step 6): Uncomment here to add Pusher to the pipeline.
  # components.append(pusher)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # Change this value to control caching of execution results. Default value
      # is `False`.
      #enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
