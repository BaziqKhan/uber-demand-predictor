import pandas as pd
from pathlib import Path

def read_data(path):
    return pd.read_csv(path,parse_dates=['15_minutes_bin'])


if __name__ == "__main__": 
    curr_path = Path(__file__)
    root_dir = curr_path.parent.parent.parent
    
    df = read_data(root_dir / 'data' / 'processed' / 'features.csv')
    
    df = df.sort_values(by=['region','15_minutes_bin'])
    
    for lag in [1,2,3,4]:
        df[f'lag_{lag}'] = df.groupby('region')['total_pickups'].shift(lag)
        
    df = df.dropna()
    
    df['month'] = df['15_minutes_bin'].dt.month
    df['day of week'] = df['15_minutes_bin'].dt.dayofweek
    
    df.set_index('15_minutes_bin', inplace=True)
    
    train_data = df.loc[df['month'].isin([1,2]),:]
    test_data = df.loc[df['month'] == 3,:]
    
    train_data = train_data.drop(columns=['month'])
    test_data = test_data.drop(columns=['month'])
    
    train_data.to_csv(root_dir / 'data' / 'processed' / 'train_data.csv', index=True)
    test_data.to_csv(root_dir / 'data' / 'processed' / 'test_data.csv', index=True)