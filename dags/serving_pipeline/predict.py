import mlflow
from mlflow import MlflowClient, pyfunc
import pandas as pd
from datetime import datetime, timedelta
import nfl_data_py as nfl
import holidays
import openmeteo_requests
import requests_cache
from retry_requests import retry
import streamlit as st

import numpy as np
import sys
sys.path.append('/Users/shawarnawaz/PycharmProjects/bart_rider_forecasting/dags')
from model_creation.get_data_from_hopsworks import get_feature_view_data
from model_creation.model_variables import *
from data_creation.helper_functions import get_temporal_features
from data_creation.feature_store_variables import *
from variables import *
from visualization.data_visualization import plot_line_plot
import logging
logging.basicConfig(level=logging.INFO)


def get_model(experiment_name, client):
    logging.info('Comparing current model with previous best model.')
    prev_best_model_version = client.get_model_version_by_alias(experiment_name, 'Staging')
    prev_run_id = prev_best_model_version.run_id
    prev_run_version = int(prev_best_model_version.version)
    logging.info(f'Loading Previous model from Run ID -> {prev_run_id} and Run Version -> {prev_run_version}')
    prev_best_model = pyfunc.load_model(f'runs:/{prev_run_id}/models')
    logging.info('Model Load Complete!')
    return prev_best_model

def click_button():
    st.session_state.clicked = True

def choose_start_end_time(start, end, calc_placeholder):
    # date_start, date_end = None, None
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    date_end = st.sidebar.date_input(
                    label= 'Insert Date upto which you want to forecast the Number of BART Riders for',
                    value= None,
                    min_value = start,
                    max_value = end,
                    format='YYYY-MM-DD',
                    )
    if date_end:
        button = st.sidebar.button('Estimate Passengers', on_click=click_button)
        if st.session_state.clicked:
            st.session_state.clicked = True
            calc_placeholder.write('Calculating....')
    return pd.to_datetime(date_end)

if __name__ == '__main__':
    startup_placeholder = st.empty()
    calc_placeholder = st.sidebar.empty()
    startup_placeholder.write('Please wait while console loads.')
    logging.info(f'Initializing client: {tracking_uri}')
    client = MlflowClient(tracking_uri=tracking_uri)
    model = get_model(experiment_name=experiment_name, client=client)
    fv, data, _ = get_feature_view_data(fs_project_name=feature_store_projectid,
                                        fs_apikey=hopsworks_apikey,
                                        fv_name=feature_view_name,
                                        fv_version=1,
                                        training_dataset_version=1)
    data['date'] = pd.to_datetime(data['date']).apply(lambda x: x.replace(tzinfo=None))
    data.sort_values('date', inplace=True)
    max_date = data['date'].max()
    min_req_date = max_date - timedelta(days=14)
    start = datetime.now() + timedelta(days=1)
    end = pd.to_datetime('2024-02-18')
    startup_placeholder.empty()
    end_date = choose_start_end_time(start, end, calc_placeholder)
    if end_date:
        int_df = data[(data['date'] >= min_req_date) & (data['date'] <= max_date)]
        int_df.set_index('date', inplace=True)
        int_df.to_csv('orig.csv')
        req_start_date = max_date + timedelta(days=1)
        date_range = pd.date_range(req_start_date, end_date)
        date_range = [pd.to_datetime(date) for date in date_range]
        years_req = list(range(req_start_date.year - 1, end_date.year + 2))
        nfl_data = list(pd.to_datetime(nfl.import_schedules(years_req).query('home_team == "SF"')['gameday']))
        hol_dates = sorted(list(holidays.UnitedStates(years=years_req).keys()))
        hol_dates = list(map(lambda x: pd.to_datetime(x), hol_dates))
        default_days_to_gd = 266

        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        lat, longitude = [37.773972, -122.431297]
        updated_rolling_features = [f'{col}_two_week_avg' for col in rolling_avg_features]
        feats = [feat for feat in model_features if feat != target_column]

        params = {
            "latitude": lat,
            "longitude": longitude,
            "start_date": str(req_start_date.date()),
            "end_date": str(end_date.date()),
            "daily": list(weather_params_req.keys())
        }
        responses = openmeteo.weather_api(weather_url, params=params)
        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        daily = response.Daily()
        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )}
        for idx, param in enumerate(weather_params_req):
            daily_value = daily.Variables(idx).ValuesAsNumpy()
            daily_data[weather_params_req[param]] = daily_value
        daily_data = pd.DataFrame(daily_data)
        daily_data.set_index('date', inplace = True)
        int_df = pd.concat([int_df, daily_data], axis = 0)
        for date in date_range:
            int_df.loc[date, 'year'], int_df.loc[date, 'month'], int_df.loc[date, 'day'], int_df.loc[date, 'day_of_week'], int_df.loc[date, 'is_weekend'] = get_temporal_features(date)

            next_hol = list(filter(lambda x: x >= date, hol_dates))
            if next_hol:
                next_hol = next_hol[0]
                int_df.loc[date, 'days_to_next_hol'] = (next_hol - date).days

            prev_hol = list(filter(lambda x: x <= date, hol_dates))
            if prev_hol:
                prev_hol = prev_hol[-1]
                int_df.loc[date, 'days_since_hol'] = (date - prev_hol).days

            next_gd = list(filter(lambda x: x >= date, nfl_data))
            if next_gd:
                next_gd = next_gd[0]
                int_df.loc[date, 'days_to_next_gd'] = (next_gd - date).days
            else:
                int_df.loc[date, 'days_to_next_gd'] = default_days_to_gd
                default_days_to_gd -= 1

            prev_gd = list(filter(lambda x: x <= date, nfl_data))
            if prev_gd:
                prev_gd = prev_gd[-1]
                int_df.loc[date, 'days_since_gd'] = (date - prev_gd).days

            if date == next_hol:
                int_df.loc[date, 'is_hol'] = 1
            else:
                int_df.loc[date, 'is_hol'] = 0

            if date == prev_gd:
                int_df.loc[date, 'is_gd'] = 1
            else:
                int_df.loc[date, 'is_gd'] = 0

            int_df.loc[date, updated_rolling_features] = list(int_df.loc[date - timedelta(days=14):date - timedelta(days=1), rolling_avg_features].mean())
            int_df.loc[date, target_column] = model.predict(np.array(int_df.loc[date, feats]).reshape(1,-1))

        plot_line_plot(int_df[[target_column]], req_start_date.strftime('%Y-%m-%d'))
        calc_placeholder.empty()