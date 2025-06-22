import dask.dataframe as dd
from pathlib import Path

# inlier range for latitude and longitude
min_latitude = 40.60
max_latitude = 40.85
min_longitude = -74.05
max_longitude = -73.70

#inlier tange for fare_amount and trip_distance
min_fare_amount_val = 0.50
max_fare_amount_val = 81.0
min_trip_distance_val = 0.25
max_trip_distance_val = 24.43


def read_dask_df(path):
    return dd.read_csv(path,usecols=["tpep_pickup_datetime","trip_distance","pickup_longitude",
                              "pickup_latitude","dropoff_longitude",
                              "dropoff_latitude","fare_amount"],
                              parse_dates=["tpep_pickup_datetime"])

def remove_outliers(df):
    df = df.loc[(df["pickup_latitude"].between(min_latitude,max_latitude,inclusive="both")) &
                (df["pickup_longitude"].between(min_longitude,max_longitude,inclusive="both")) &
                (df["dropoff_longitude"].between(min_longitude,max_longitude,inclusive="both")) &
                (df["dropoff_latitude"].between(min_latitude,max_latitude,inclusive="both")) &
                (df["fare_amount"].between(min_fare_amount_val,max_fare_amount_val,inclusive="both")) &
                (df["trip_distance"].between(min_trip_distance_val,max_trip_distance_val,inclusive="both")),:]
    
    df = df.drop(columns=["dropoff_latitude","dropoff_longitude","trip_distance","fare_amount"])

    df = df.compute()
    return df



curr_path = Path(__file__)
root_dir = curr_path.parent.parent.parent
data_path = root_dir/"data"/"raw"

df_name = ["yellow_tripdata_2016-01.csv","yellow_tripdata_2016-02.csv",
               "yellow_tripdata_2016-03.csv"]
    
dfs = []
    
for name in df_name:
    data = read_dask_df(data_path/name)
    dfs.append(data)
        
df_final = dd.concat(dfs,axis=0) 
df_final = remove_outliers(df_final)  
    
df_final.to_csv(root_dir/"data"/'interim'/"cleaned_data.csv",index=False)