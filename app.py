import pandas as pd
import joblib 
from pathlib import Path
from sklearn.pipeline import Pipeline
from time import sleep
import mlflow
from mlflow.client import MlflowClient
import streamlit as st
import datetime as dt

import dagshub
dagshub.init(repo_owner='baziq12', repo_name='uber-demand-predictor', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/baziq12/uber-demand-predictor.mlflow')

register_model_name = 'uber_demand_prediction'
stage = 'Production'
model_uri = f'models:/{register_model_name}/{stage}'

model = mlflow.sklearn.load_model(model_uri=model_uri)

root_dir = Path(__file__).parent

plot_data_path = root_dir/'data'/'external'/'plot_data.csv'
data_path = root_dir/'data'/'processed'/'test_data.csv'

kmeans_path = root_dir/'models'/'kmeans.joblib'
transformer_path = root_dir/'models'/'transformer.joblib'
scaler_path = root_dir/'models'/'scaler.joblib'

scaler = joblib.load(scaler_path)
kmeans = joblib.load(kmeans_path)
transformer = joblib.load(transformer_path)

df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=['15_minutes_bin'],index_col='15_minutes_bin')

st.title('Uber Demand in NYC')
st.sidebar.title('Options')
map_type = st.sidebar.radio(label='Select the Type of Map'
                            ,options = ['Complete NYC Map','Only for Neigbourhood Regions']
                            ,index = 1)

st.subheader('Date')
date = st.date_input(label='Select Date',value = None, min_value=dt.date(year = 2016,month = 3,day = 1)
                     , max_value=dt.date(year = 2016,month = 3,day = 31))

st.write('Date Selected:',date)

st.subheader('Time')
time = st.time_input(label='Select Time'
                     ,value=None)

st.write('Time Selected:',time)

if date and time:
    delta = dt.timedelta(minutes=15)
    next_interval = dt.datetime(
        year = date.year,
        month= date.month,
        day = date.day,
        hour= time.hour,
        minute= time.minute
    ) + delta
    st.write('Demand for Time:**',next_interval.time())

    index = pd.Timestamp(f'{date} {next_interval.time()}')
    st.write('Date & Time:',index)

        
    st.subheader('Location')
    sample_loc = df_plot.sample(1).reset_index(drop=True)
    latitude =sample_loc['pickup_latitude'].item()
    longitude = sample_loc['pickup_longitude'].item()

    region = sample_loc['region'].item()
    st.write('Your current location:\n')
    st.write('Latitude:',latitude)
    st.write('Longitude:',longitude)

    with st.spinner('Fetching your current region'):
        sleep(3)
        
    st.write('Region_ID:',region)

    scaled_cord = scaler.transform(sample_loc[['pickup_longitude','pickup_latitude']])

    st.subheader('MAP')

    colors = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", 
                "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00", 
                "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", 
                "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", 
                "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347", 
                "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"]

    region_colors = {region:colors[i] for i,region in enumerate(df['region'].unique().tolist())}

    df_plot['color'] = df_plot['region'].map(region_colors)
    
    pipe = Pipeline([
        ('encoder',transformer),
        ('linear_reg',model)
    ])
    
    if map_type == 'Complete NYC Map':
        
        progress_bar = st.progress(
            value = 0,
            text = 'Operarion in Progress...'
        )
        
        for percent in range(100):
            progress_bar.progress(
                value = percent + 1,
                text = f'Operation in Progress...{percent + 1}%'
            )
        st.map(
            data = df_plot,
            latitude= 'pickup_latitude',
            longitude= 'pickup_longitude',
            size = 0.01,
            color='color'
        )
        progress_bar.empty()
        input_data = df.loc[index,:].sort_values('region')
        
        target = df['total_pickups']
        
        prediction = pipe.predict(input_data.drop(columns=['total_pickups']))
        
        st.markdown('Map Legend')
        for idx in range(0,30):
            color = colors[idx]
            demand = prediction[idx]
            if region == idx:
                region_id = f'{idx} (current_region)'
            else:
                region_id = idx
            
            st.markdown(
                     f'<div style="display: flex; align-items: center;">'
                     f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                     f'Region ID: {region_id} <br>'
                     f"Demand: {int(demand)} <br> <br>", unsafe_allow_html=True
                     )
        
    elif map_type == 'Only for Neigbourhood Regions':
        
        distances = kmeans.transform(scaled_cord).ravel().tolist()
        distances = list(enumerate(distances,start = 0))
        
        sorted_distances = sorted(distances,key=lambda x: x[1])[0:9]
        desired_region = sorted([idx[0] for idx in sorted_distances])
        
        df_plot_filtered = df_plot.loc[df_plot['region'].isin(desired_region),:]
        
        progress_bar = st.progress(
            value = 0,
            text = 'Operation in Progress. Please Wait'
        )
        for percent in range(100):
            progress_bar.progress(value=percent+1,
                                  text='Operation in Progress. Please Wait')
            
        
        st.map(
            data = df_plot_filtered,
            latitude = 'pickup_latitude',
            longitude = 'pickup_longitude',
            color = 'color'
        )    
        progress_bar.empty()
        
        input_data = df.loc[index,:]
        input_data = input_data.loc[input_data['region'].isin(desired_region)].sort_values('region')
        
        target = input_data['total_pickups']
        
        prediction = pipe.predict(input_data.drop(columns=['total_pickups']))
        
        
        st.markdown('Map Legend')
        for idx in range(0,9):
            color = region_colors[desired_region[idx]]
            demand = prediction[idx]
            if region == desired_region[idx]:
                region_id = f'{desired_region[idx]} (current_region)'
            else:
                region_id = desired_region[idx]
            
            st.markdown(
                     f'<div style="display: flex; align-items: center;">'
                     f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                     f'Region ID: {region_id} <br>'
                     f"Demand: {int(demand)} <br> <br>", unsafe_allow_html=True
                     )
