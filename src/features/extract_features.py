import pandas as pd
import joblib
from sklearn.cluster import MiniBatchKMeans
import yaml
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_data(path):
    return pd.read_csv(path,chunksize=100000,usecols=['pickup_latitude','pickup_longitude'])

def save_model(model, path):
    joblib.dump(model, path)
    
def read_params(path):
    with open(path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def train_model(root_dir,scaler):
    
    data = load_data(root_dir/'data'/'interim'/'cleaned_data.csv')
    
    params = read_params(root_dir/'params.yaml')
    
    kmeans = MiniBatchKMeans(**params['extract_features']['minibatchkmeans'])
    
    for chunk in data:
        scaled_chunk = scaler.transform(chunk)
        kmeans.partial_fit(scaled_chunk)
        
    return kmeans
    
if __name__ == "__main__":
    curr_path = Path(__file__)
    root_dir = curr_path.parent.parent.parent
    scaler = StandardScaler()
    data_path = root_dir / 'data' / 'interim'/ 'cleaned_data.csv'
    data = load_data(data_path)
    
    for chunk in data:
        scaler.partial_fit(chunk)
    
    #Saving Scaler
    save_model(scaler,root_dir/'models'/'scaler.joblib')
    
    kmeans = train_model(root_dir, scaler)
    
    save_model(kmeans, root_dir/'models'/'kmeans.joblib')
    
    df = pd.read_csv(data_path,parse_dates=['tpep_pickup_datetime'])
    location_subset = df.loc[:,['pickup_longitude','pickup_latitude']]
    df['region'] = kmeans.predict(scaler.transform(location_subset))
    
    df = df.drop(columns=['pickup_latitude','pickup_longitude'])
    
    df['15_minutes_bin'] = df['tpep_pickup_datetime'].dt.floor('15T')
    
    df = df.groupby(['15_minutes_bin','region']).size().reset_index(name='total_pickups')
    
    params = read_params(root_dir/'params.yaml')
    
    df['avg_pickups'] = df.groupby('region')['total_pickups'].ewm(**params['extract_features']['ewm']).mean().round().values
    
    df.to_csv(root_dir/'data'/'processed'/'features.csv',index=False)