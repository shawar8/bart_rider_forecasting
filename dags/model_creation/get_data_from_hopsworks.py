import hopsworks
import sys
sys.path.append('/Users/shawarnawaz/PycharmProjects/bart_rider_forecasting/dags')

from datetime import timedelta
from typing import Tuple, List
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
def get_feature_view_data(fs_apikey: str,
                          fs_project_name: str,
                          fv_name: str,
                          fv_version: int,
                          training_dataset_version: int):
    logging.info('Logging into Hopsworks')
    project= hopsworks.login(project=fs_project_name,
                             api_key_value=fs_apikey)
    logging.info('Getting Feature Store')
    fs= project.get_feature_store()
    logging.info('Getting Feature View')
    fv= fs.get_feature_view(name= fv_name,
                            version= fv_version)
    logging.info('Getting Training Data')
    data, _ = fv.get_training_data(training_dataset_version=training_dataset_version)
    logging.info('Creating Metadata')
    md = fv.to_dict()
    md['query'] = md['query'].to_string()
    md['features'] = [feature.name for feature in md['features']]
    md['link']= fv._feature_view_engine._get_feature_view_url(fv)
    md['fv_version']= fv_version
    md['training_data_version'] = training_dataset_version
    md['fs_project_name'] = fs_project_name
    md['fv_name'] = fv_name
    return fv, data, md

def split_train_val(data: pd.DataFrame,
                    num_months_val: int,
                    target_column: str,
                    model_features: List)-> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data['date'] = pd.to_datetime(data['date'])
    # Getting the number of months we want to use as a test set
    last_date= data['date'].max()
    validation_thresh_date = last_date - timedelta(days= 30 * num_months_val)
    logging.info('Splitting the data into Train and Test')
    X_train = data[data['date'] < validation_thresh_date]
    X_test = data[data['date'] >= validation_thresh_date]
    X_train = X_train[model_features]
    X_test = X_test[model_features]
    # Creating the labels
    y_train = X_train.pop(target_column)
    y_test = X_test.pop(target_column)
    return X_train, y_train, X_test, y_test
