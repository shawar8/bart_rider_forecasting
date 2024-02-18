import pandas as pd
import hopsworks
from great_expectations.core import ExpectationSuite
import hsfs
from hsfs.feature_group import FeatureGroup
from data_creation.data_validation import build_expectations_suite
from data_creation.feature_store_variables import *
import variables
import logging
logging.basicConfig(level=logging.INFO)


def upload_to_hopsworks(df: pd.DataFrame,
                        validation_expectation_suite: ExpectationSuite,
                        feature_group_version: int):
    '''Steps: 1) Login to feature store with api key and project name
        2) get feature store
        3) create feature group
        4) upload the data
        5) upload feature descriptions
        6) get data statistics'''

    logging.info('Logging into Hopsworks')
    project = hopsworks.login(api_key_value=hopsworks_apikey,
                              project=feature_store_projectid)

    logging.info('Creating Feature Store')
    feature_store=project.get_feature_store()
    ##############################################
    # logging.info('Deleting existing Feature group')
    # fgroup = feature_store.get_or_create_feature_group(name='BART_riders_prediction',
    #                                                    version=feature_group_version)
    # fgroup.delete()
    ##############################################

    logging.info('Creating Feature Group')
    feature_group = feature_store.get_or_create_feature_group(
        name=feature_group_name,
        version=1,
        description='Feature group for BART Data',
        primary_key= ['date'],
        event_time='data_input_time',
        online_enabled=False,
        expectation_suite=validation_expectation_suite
    )
    logging.info('Inserting Data into Feature Group')
    feature_group.insert(
        features=df,
        overwrite=False,
        write_options={
            'wait_for_job': True
        }
    )
    feature_description = [
        {
            "name": "date",
            "description": """
                                   Datetime interval in UTC when the data was observed.
                                   """,
            "validation_rules": "Always full hours, i.e. minutes are 00",
        },
        {
            "name": "data_input_time",
            "description": """
                                   TimeStamp for when the data was uploaded.
                                   """,
        },
    ]
    logging.info('Creating Feature Description')
    for description in feature_description:
        logging.info(f'Creating Feature Description for {description["name"]}')
        feature_group.update_feature_description(
            description['name'], description['description']
        )
    logging.info('Creating Feature Group Statistics')
    feature_group.statistics_config = {
        'enabled': True,
        'histograms': True,
        'correlations': True
    }
    logging.info('Updating Statistics Config')
    feature_group.update_statistics_config()
    logging.info('Computing Statistics')
    feature_group.compute_statistics()
    logging.info('Completed Computing Statistics')
    return feature_group

def create_feature_view(feature_group_name: str,
                            feature_group_version: int,
                            feature_view_name: str):
    project= hopsworks.login(api_key_value= hopsworks_apikey,
                             project= feature_store_projectid)
    feature_store = project.get_feature_store()
    try:
        logging.info('Getting feature views present')
        feature_views = feature_store.get_feature_views(name= feature_view_name)
    except:
        logging.info('No Feature View present')
        feature_views = []
    if feature_views:
        for feature_view in feature_views:
            try:
                feature_view.delete_all_training_datasets()
            except hsfs.client.exceptions.RestAPIError():
                logging.error(f"Failed to delete training datasets for feature view {feature_view.name} with version {feature_view.version}.")
            try:
                feature_view.delete()
            except hsfs.client.exceptions.RestAPIError:
                logging.error(f"Failed to delete feature view {feature_view.name} with version {feature_view.version}.")
    fgroup = feature_store.get_or_create_feature_group(name=feature_group_name,
                                                           version=feature_group_version)
    ds_query = fgroup.select_all()
    feature_view=feature_store.create_feature_view(
            name=feature_view_name,
            description='BART Rider Data',
            query=ds_query,
            labels=[]
        )
    logging.info(f'Creating Training Dataset')
    feature_view.create_training_data(
            description='BART Light GBM Model Training Data',
            data_format='csv',
            # start_time=start_date,
            # end_time=end_date,
            write_options={"wait_for_job": True},
            coalesce=False
    )




