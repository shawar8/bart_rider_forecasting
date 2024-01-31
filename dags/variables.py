from collections import OrderedDict
from datetime import datetime


url = 'data.sfgov.org'
weather_url = 'https://archive-api.open-meteo.com/v1/archive'
bart_data_apikey = ''

weather_params_req = OrderedDict({"temperature_2m_max": "max_temp",
                                  "temperature_2m_min": "min_temp",
                                  "precipitation_sum": "total_rain",
                                  "precipitation_hours": "rain_duration",
                                  "daylight_duration": "num_daylight_hours",
                                  "wind_speed_10m_max": "max_wind_speed"})

lat, longitude = [37.773972, -122.431297]
original_target_column = '_16'
target_column = 'num_passengers'
features_required = ['date', 'day_of_week', '_16', 'is_gameday', 'days_since_hol',
                     'days_to_next_hol', 'days_since_gd', 'days_to_next_gd',
                     'which_day', 'day', 'month', 'is_weekend']
rolling_avg_features = [target_column, 'max_temp', 'min_temp', 'total_rain', 'rain_duration',
                               'num_daylight_hours', 'max_wind_speed']
current_year = datetime.now().year

model_features = ['date', 'day_of_week', target_column, 'is_gameday', 'days_since_hol',
                  'days_to_next_hol', 'days_since_gd', 'days_to_next_gd', 'max_temp',
                  'min_temp', 'total_rain', 'rain_duration', 'num_daylight_hours',
                  'max_wind_speed', f'{target_column}_two_week_avg',
                  'max_temp_two_week_avg', 'min_temp_two_week_avg',
                  'total_rain_two_week_avg', 'rain_duration_two_week_avg',
                  'num_daylight_hours_two_week_avg', 'max_wind_speed_two_week_avg',
                  'month', 'day', 'which_day', 'is_weekend']

num_months_val = 4


model_parameters = {
        "max_depth": [3, 4, 5, 6, 10],
        "num_leaves": [10, 20, 30, 40, 100],
        "learning_rate": [0.01, 0.05],
        "n_estimators": [50, 100, 300],
        "colsample_bytree": [0.5, 0.7],
        "verbose": [-1],
        "n_jobs": [-1],
        "random_state": [42]
    }

config = {
  "dest_bucket": "bart-passenger-prediction", # This will be auto created
  "minio_endpoint": "",
  "minio_username": "",
  "minio_password": "",
}
