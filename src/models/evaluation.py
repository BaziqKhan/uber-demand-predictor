import mlflow.models
import pandas as pd
import joblib
from pathlib import Path
import mlflow
import sklearn.metrics as mt
import json
import dagshub

dagshub.init(repo_owner='baziq12', repo_name='uber-demand-predictor', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/baziq12/uber-demand-predictor.mlflow')
mlflow.set_experiment('DVC Pipeline')

def save_run_information(file_path,run_id,artifact_path,model_uri):
    run_information = {
        'run_id':run_id,
        'artifact_path':artifact_path,
        'model_uri':model_uri
    }
    
    with open(file_path,'w') as file:
        json.dump(run_information,file,indent = 4)
    
    
if __name__ == '__main__':
    curr_path = Path(__file__)
    root_dir = curr_path.parent.parent.parent
    data_path = root_dir / 'data' / 'processed' 
    test_df = pd.read_csv(data_path/ 'test_data.csv', parse_dates=['15_minutes_bin'], index_col='15_minutes_bin')
    
    xtest = test_df.drop(columns=['total_pickups'])
    ytest = test_df['total_pickups']
    
    transformer = joblib.load(root_dir/'models'/'transformer.joblib')
    
    xtest_encoded = transformer.transform(xtest)
    
    model = joblib.load(root_dir/'models'/'model.joblib')
    
    
    ypred = model.predict(xtest_encoded)
    
    loss = mt.mean_absolute_percentage_error(ytest,ypred)
    
    with mlflow.start_run(run_name='model'):
        mlflow.log_params(model.get_params())
        
        mlflow.log_metric('MAPE',loss)
        
        training_data = mlflow.data.from_pandas(pd.read_csv(data_path/'train_data.csv',parse_dates=['15_minutes_bin'],index_col='15_minutes_bin'),targets = 'total_pickups')
        
        validation_data = mlflow.data.from_pandas(pd.read_csv(data_path/'test_data.csv',parse_dates=['15_minutes_bin'],index_col='15_minutes_bin'),targets = 'total_pickups')
        
        mlflow.log_input(training_data,context='training')
        mlflow.log_input(validation_data,context='validation')
        
        model_signature = mlflow.models.infer_signature(xtest_encoded,ypred)
        
        logged_model = mlflow.sklearn.log_model(model,'demand_prediction',
                                                signature=model_signature,pip_requirements='requirements.txt')
        
        
    run_id = logged_model.run_id
    artifact_path = logged_model.artifact_path
    model_uri = logged_model.model_uri
    
    json_file_save_path = root_dir / 'run_information.json'
    
    save_run_information(json_file_save_path,run_id,artifact_path,model_uri)
        
    

 