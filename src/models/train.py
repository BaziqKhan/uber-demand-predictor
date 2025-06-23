from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def load_data(path):
    return pd.read_csv(path, parse_dates=['15_minutes_bin'], index_col='15_minutes_bin')

def save_model(model, path):
    joblib.dump(model, path)


if __name__ == "__main__":
    curr_path = Path(__file__)
    root_dir = curr_path.parent.parent.parent

    train_data = load_data(root_dir / 'data' / 'processed' / 'train_data.csv')
    
    X = train_data.drop(columns=['total_pickups'])
    y = train_data['total_pickups']
    
    transformer = ColumnTransformer(transformers = [
        ('encoder',OneHotEncoder(drop = 'first',sparse_output=False,handle_unknown='ignore'),['region','day of week'])],
        remainder='passthrough',n_jobs=-1)
    
    transformer.fit(X)
    save_model(transformer, root_dir / 'models' / 'transformer.joblib')
    
    X_encoded = transformer.transform(X)
    model = LinearRegression()
    
    model.fit(X_encoded, y)
    
    model_path = root_dir / 'models' / 'model.joblib'
    save_model(model, model_path)