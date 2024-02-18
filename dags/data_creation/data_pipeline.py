# import sys
# sys.path.append('.')
# from variables import *
# import data_creation.helper_functions
# import json
# from minio import Minio
# import pandas as pd
# import numpy as np
# import os

from airflow.decorators import task, dag
from airflow.utils.dates import days_ago

###### additional packages #######

###################################

import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.INFO)


@dag(dag_id='Data_Pipeline',
     description='Read Data',
     default_args={'start_date': days_ago(0)})
def get_bart_data():
    @task.virtualenv(task_id='read_data_bart',
                     requirements=['sodapy', 'pandas', 'numpy', 'apache-airflow',
                                   'holidays', 'nfl_data_py', 'minio'])
    def read_data():
        from airflow.models import Variable
        from sodapy import Socrata
        from datetime import datetime

        from data_creation.helper_functions import get_temporal_features, dump_data
        from variables import original_target_column, target_column
        import pandas as pd
        import logging
        logging.basicConfig(level=logging.INFO)

        logging.info('Start')
        url = Variable.get('url')
        app_token = Variable.get('sfgov_token')
        logging.info(url)
        logging.info(app_token)
        current_year=datetime.now().year
        stop = Variable.get('stop')
        logging.info('Starting to read data')
        client = Socrata(url, app_token=app_token)

        df = pd.DataFrame(client.get('m2xz-p7ja', limit=10000000))
        logging.info('Data Read Successfully')
        df['date'] = pd.to_datetime(df['date'])
        df[stop] = df[stop].astype(float)
        years_req = list(range(current_year-3, current_year+1))

        dump_data(df, 'variables.pkl', 'variables.pkl')
        logging.info('Creating Date Features')
        df['year'], df['month'], df['day'], df['day_of_week'], df['is_weekend'] = zip(*pd.Series(df['date']).apply(get_temporal_features))
        logging.info('Date Features Created')
        df = df[df['year'].isin(years_req)]
        df = df[pd.notnull(df[stop])]
        logging.info(f'Shape of dataframe -> {df.shape}')
        df.reset_index(drop=True, inplace = True)
        df.rename(columns={original_target_column: target_column},
                  inplace=True)

        filename = 'bart_data.csv'
        dump_data(df, filename, filename)
        logging.info(years_req)
        data_req = {'data_path': filename}
        dump_data(data_req, 'variables.pkl', 'variables.pkl')

    @task.virtualenv(task_id='get_nfl_data',
                     requirements=['nfl-data-py', 'pandas', 'numpy', 'apache-airflow',
                                   'holidays', 'minio'])
    def get_nfl_data():
        import nfl_data_py as nfl
        import pandas as pd
        import logging
        logging.basicConfig(level=logging.INFO)
        from datetime import datetime
        from data_creation.helper_functions import dump_data, get_data_from_minio

        data_req = get_data_from_minio('variables.pkl')
        current_year= datetime.now().year
        years_req = list(range(current_year-4, current_year+1))
        nfl_data = nfl.import_schedules(years_req).query('home_team == "SF"')[['gameday']]
        nfl_data['gameday'] = pd.to_datetime(nfl_data['gameday'])
        nfl_data['is_gameday'] = 1
        filename = 'nfl_data.csv'
        dump_data(nfl_data, filename, filename)
        data_req['nfl_data'] = filename
        dump_data(data_req, 'variables.pkl', 'variables.pkl')

    @task.virtualenv(task_id='calculate_hols_gamedays',
                     requirements=['pandas', 'holidays', 'nfl-data-py', 'numpy',
                                   'minio'])
    def calc_hols_and_gamedays():
        import holidays
        import pandas as pd
        from datetime import datetime
        from data_creation.helper_functions import get_days_count, dump_data, get_data_from_minio
        import logging
        logging.basicConfig(level=logging.INFO)

        curr_year = datetime.now().year
        years_req = list(range(curr_year - 4, curr_year + 1))
        data_req = get_data_from_minio('variables.pkl')
        data_path = data_req['data_path']
        nfl_path = data_req['nfl_data']
        data = get_data_from_minio(data_path)
        nfl_data = get_data_from_minio(nfl_path)
        data['date'] = pd.to_datetime(data['date'])
        gamedays = sorted(list(pd.to_datetime(nfl_data['gameday'])))
        holidays = sorted(list(holidays.UnitedStates(years=years_req).keys()))
        holidays = list(map(lambda x: pd.to_datetime(x), holidays))

        data = get_days_count(data, 'hol', holidays)
        data = get_days_count(data, 'gd', gamedays)

        dump_data(data, 'bart_data.csv', 'bart_data.csv')

    @task.virtualenv(task_id='get_weather_data',
                     requirements=['pandas', 'holidays', 'nfl-data-py', 'numpy',
                                   'minio', 'requests_cache', 'retry_requests',
                                   'openmeteo_requests'])
    def get_weather_data():
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry
        import pandas as pd
        from data_creation.helper_functions import dump_data, get_data_from_minio
        from variables import features_required, weather_params_req, weather_url
        from datetime import timedelta
        import logging
        logging.basicConfig(level=logging.INFO)

        data_req = get_data_from_minio('variables.pkl')
        data_path = data_req['data_path']
        data = get_data_from_minio(data_path)
        data['date'] = pd.to_datetime(data['date'])
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        lat, longitude = [37.773972, -122.431297]
        min_date = data['date'].min() - timedelta(days=30)
        max_date = data['date'].max()

        params = {
                "latitude": lat,
                "longitude": longitude,
                "start_date": str(min_date.date()),
                "end_date": str(max_date.date()),
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
        data = data.merge(daily_data, how='left', on='date')
        final_features = features_required + list(weather_params_req.values())

        dump_data(data, data_path, data_path)
        dump_data(final_features, 'final_features.pkl', 'final_features.pkl')

    @task.virtualenv(task_id='get_rolling_features',
                     requirements=['pandas', 'holidays', 'nfl-data-py', 'numpy',
                                   'minio'])
    def get_rolling_features():
        from data_creation.helper_functions import dump_data, get_data_from_minio
        from datetime import datetime, timedelta
        import pandas as pd
        from variables import rolling_avg_features

        data_req = get_data_from_minio('variables.pkl')
        data_path = data_req['data_path']
        data = get_data_from_minio(data_path)
        data['date'] = pd.to_datetime(data['date'])
        one_week_avg = data[rolling_avg_features].shift(1).rolling(7).mean()
        two_week_avg = data[rolling_avg_features].shift(1).rolling(14).mean()
        three_week_avg = data[rolling_avg_features].shift(1).rolling(21).mean()
        one_week_sum = data[rolling_avg_features].shift(1).rolling(7).sum()
        for tbl in ['one_week_avg', 'two_week_avg', 'three_week_avg', 'one_week_sum']:
            tbl_data = locals()[tbl]
            tbl_data.columns = [f'{col}_{tbl}' for col in tbl_data.columns]
            data = pd.concat([data, tbl_data], axis=1)
        date_thresh = str((datetime.now() - timedelta(days=365 * 3)).date())
        data = data[data['date'] >= date_thresh]
        # data.set_index('date', inplace=True)
        dump_data(data, data_path, data_path, is_index=True)

    @task.virtualenv(task_id='upload_data_to_hopsworks',
                     requirements=['hsfs', 'hopsworks', 'great_expectations',
                                   'pandas', 'numpy', 'holidays', 'nfl-data-py',
                                   'minio'])
    def upload_data_to_hopsworks():
        from data_creation.create_feature_store import upload_to_hopsworks, create_feature_view
        from data_creation.helper_functions import get_data_from_minio
        from data_creation import feature_store_variables
        from data_creation.data_validation import build_expectations_suite
        from datetime import datetime, timedelta
        import pandas as pd
        import logging
        logging.basicConfig(level=logging.INFO)

        logging.info('Fetching Data')
        data_req = get_data_from_minio('variables.pkl')
        data_path = data_req['data_path']
        data = get_data_from_minio(data_path)
        end_date = datetime.now()
        data['date'] = pd.to_datetime(data['date'])
        data['data_input_time'] = end_date
        data= data[feature_store_variables.features]

        start_date = end_date - timedelta(days=3*365)
        data = data[(data['date'] <= end_date) & (data['date'] >= start_date)]

        logging.info('Building Expectation Suite')
        expectation_suite = build_expectations_suite()
        logging.info('Uploading data to Hopsworks')
        logging.info(f'Size of dataframe: {data.shape}')
        upload_to_hopsworks(df=data, validation_expectation_suite=expectation_suite,
                            feature_group_version=1)
        logging.info('Creating Feature View')
        create_feature_view(feature_group_name= feature_store_variables.feature_group_name,
                            feature_group_version= 1,
                            feature_view_name= 'bart_training_data'
                            )



    read_data() >> get_nfl_data() >> calc_hols_and_gamedays() >> get_weather_data() >> get_rolling_features() >> upload_data_to_hopsworks()

get_bart_data()