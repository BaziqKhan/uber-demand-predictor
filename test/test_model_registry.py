import pytest
import mlflow
from mlflow.client import MlflowClient
import dagshub

dagshub.init(repo_owner='baziq12', repo_name='uber-demand-predictor', mlflow=True)


mlflow.set_tracking_uri('https://dagshub.com/baziq12/uber-demand-predictor.mlflow')

model_name = 'uber_demand_prediction'

stage = 'Staging'
@pytest.mark.parametrize(argnames="model_name,stage",
                         argvalues=[(model_name,stage)])

def test_load_model_from_registry(model_name,stage):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name,stages=[stage])
    latest_version = latest_versions[0].version if latest_versions else None
    assert latest_versions is not None, f"No model at {stage} stage"

    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    assert model is not None, f"Failed to load model from registry"
    print(f"The {model_name} model with version {latest_version} was loaded successfully")
    
    
    