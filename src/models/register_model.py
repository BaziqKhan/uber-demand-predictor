import dagshub
import mlflow
from mlflow.client import MlflowClient 
from pathlib import Path
import json

dagshub.init(repo_owner='baziq12', repo_name='uber-demand-predictor', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/baziq12/uber-demand-predictor.mlflow')

with open('run_information.json', 'r') as file:
    model_info = json.load(file)

model_uri = model_info['model_uri']

version = mlflow.register_model(model_uri=model_uri, name='uber_demand_prediction')

stage = 'Staging'

client = MlflowClient()
client.transition_model_version_stage(
    name='uber_demand_prediction',
    version=version.version,
    stage=stage,
    archive_existing_versions=False
)