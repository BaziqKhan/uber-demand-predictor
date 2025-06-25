import mlflow
from mlflow.client import MlflowClient

import dagshub
dagshub.init(repo_owner='baziq12', repo_name='uber-demand-predictor', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/baziq12/uber-demand-predictor.mlflow')

model_name = 'uber_demand_prediction'

stage = 'Staging'

promote_stage = 'Production'

client = MlflowClient()

latest_version = client.get_latest_versions(model_name,stages=[stage])[0].version

client.transition_model_version_stage(
    name = model_name,
    version=latest_version,
    stage=promote_stage,
    archive_existing_versions=True
)

