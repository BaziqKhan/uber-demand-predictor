import pytest
import joblib
import mlflow
from mlflow.client import MlflowClient
from pathlib import Path
from sklearn.pipeline import Pipeline
import pandas as pd
import sklearn.metrics as mt

import dagshub

dagshub.init(repo_owner='baziq12', repo_name='uber-demand-predictor', mlflow=True)


mlflow.set_tracking_uri('https://dagshub.com/baziq12/uber-demand-predictor.mlflow')

model_name = 'uber_demand_prediction'

stage = 'Staging'

current_path = Path(__file__)

root_dir = current_path.parent.parent

train_data_path = root_dir/'data'/'processed'/'train_data.csv'
test_data_path = root_dir/'data'/'processed'/'test_data.csv'

transformer = joblib.load(root_dir/'models'/'transformer.joblib')
client = MlflowClient()
model_version = client.get_latest_versions(model_name, stages=[stage])[0].version
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri=model_uri)

pipe = Pipeline([
    ("encoder",transformer),
    ('lin_reg',model)
])

@pytest.mark.parametrize(
    argnames='data_path,threshold',
    argvalues=[(train_data_path,0.3),
               (test_data_path,0.3)]
)

def test_performance(data_path,threshold):
    data = pd.read_csv(data_path,parse_dates=['15_minutes_bin'],index_col='15_minutes_bin')
    
    X = data.drop(columns='total_pickups')
    y = data['total_pickups']
    
    ypred = pipe.predict(X)
    
    error = mt.mean_absolute_percentage_error(y,ypred)
    print(error)
    assert error <= threshold, f'The model does not pass the performance threshold of {threshold}'
    
    