import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient
from model_variables import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import optuna
import sys
sys.path.append('/Users/shawarnawaz/PycharmProjects/bart_rider_forecasting/dags')
from data_creation.feature_store_variables import *
from model_creation.model_variables import *
from model_creation.get_data_from_hopsworks import get_feature_view_data, split_train_val
from variables import target_column, model_features

from datetime import datetime
from typing import Tuple, List, Dict
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)


def plot_residuals(model, x_test, y_test):
    # Function to Plot the residuals from model predictions

    y_pred = model.predict(x_test)
    residuals = y_test - y_pred
    fig = plt.figure()
    plt.scatter(y_test, residuals, color='red', alpha=0.5)
    plt.axhline(y=0, color='blue', linestyle='-')
    plt.xticks()
    plt.yticks()
    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig('residuals.png')
    plt.close(fig)
    return fig


def get_or_create_experiment(experiment_name, experiment_tags):
    logging.info('Checking to see if experiment exists')
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logging.info(f'Experiment {experiment_name} not found. Creating Experiment')
        mlflow.create_experiment(name=experiment_name, tags=experiment_tags)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    expid = experiment.experiment_id
    current_experiment = mlflow.set_experiment(experiment_name)
    return expid, current_experiment


def objective(trial, x_train, y_train, x_test, y_test):
    # Creating a Trial
    with mlflow.start_run(nested=True):
        # Setting the parameter ranges
        params = {'n_estimators': trial.suggest_int('n_estimators', 50, 350, step=50),
                  'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
                  'num_leaves': trial.suggest_int('num_leaves', 10, 100, step = 10),
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, step = 0.01),
                  'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7, step = 0.2),
                  'verbose': [-1],
                  'n_jobs': [-1],
                  }
        n_estim = params['n_estimators']
        # Passing the current parameters to the model
        model = lgb.LGBMRegressor(**params)
        # Fitting the model on the data
        model.fit(x_train, y_train)
        # Evaluating the model for MSE, RMSE, MAPE
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_pred, y_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
        mape = mean_absolute_percentage_error(y_pred=y_pred, y_true=y_test)
        # Setting the tags and logging the model parameters + metrics on MLFlow
        trial_name = f'mape_{mape}_rmse_{rmse}_mae_{mae}_nestim_{n_estim}'
        mlflow.set_tags(
            tags={
                'mlflow.runName': trial_name
            }
        )
        trial.set_user_attr('name', trial_name)
        mlflow.log_params(params)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mape', mape)
        mlflow.log_metric('mae', mae)
    return mape


def champion_callback(study, frozen_trial):
    # Getting the best MAPE metric so far
    global best_trial
    winner=study.user_attrs.get('winner', None)
    # Comparing the current metric with the best metric
    if study.best_value and winner != study.best_value:
        # Replacing previous best metric with current one
        study.set_user_attr('winner', study.best_value)
        # Printing the Improvement percentage
        if winner:
            best_trial= frozen_trial
            improvement_pct = (abs(winner-study.best_value)/study.best_value) * 100
            print (f'Trial {frozen_trial.number} achieved value: {frozen_trial.value} with ')
            print (f'{improvement_pct}% improvement')
        else:
            # If first trial
            print (f'Initial Trial {frozen_trial.number} achieved value {frozen_trial.value}')


def run_hyperopt(experiment_id, tag_name, run_name, study_name, model_type):
    # Creating a run and initializing a study object
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested = True):
        study = optuna.create_study(direction='minimize', study_name=study_name)
        # Passing the objective function and callback to the optimizer
        study.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test),
                       n_trials=100,
                       callbacks=[champion_callback])
        # Getting the best parameters based on the MAPE score
        params = study.best_params
        mlflow.log_params(params)
        mlflow.log_metric('best_mape', study.best_value)

        mlflow.set_tags(
            tags={
                'project': tag_name,
                'optimizer': 'optuna',
                'model_type': model_type,
                'feature_set_version': 1,
            })
    return params


if __name__ == '__main__':
    best_trial = None
    fv, data, md = get_feature_view_data(fs_project_name=feature_store_projectid,
                                         fs_apikey=hopsworks_apikey,
                                         fv_name=feature_view_name,
                                         fv_version=1,
                                         training_dataset_version=1)
    x_train, y_train, x_test, y_test = split_train_val(data, num_months_val, target_column, model_features)
    logging.info(f'Initializing client: {tracking_uri}')
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id, current_experiment = get_or_create_experiment(experiment_name, experiment_tags)
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    study_name = hyperopt_dict[experiment_name]
    logging.info('Finding best Hyperparameters')
    best_params = run_hyperopt(experiment_id=experiment_id,
                 tag_name=tag_name,
                 run_name=now,
                 study_name=study_name,
                 model_type='lightgbm')
    logging.info('Building best model')
    model = lgb.LGBMRegressor(**best_params)
    logging.info('Evaluating Best Model')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    logging.info('Calculating MAPE')
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logging.info(f'MAPE for current model: {mape}')
    logging.info('Plotting Residuals')
    fig = plot_residuals(model, x_test, y_test)
    signature = infer_signature(x_test, y_pred)
    with mlflow.start_run(experiment_id=experiment_id, run_name=now):
        mlflow.log_figure(fig, 'residuals.png')
    logging.info('Getting the Run ID for the Best trial')
    best_trial_name = best_trial.user_attrs['name']
    best_trial_id = list(mlflow.search_runs(
                         experiment_names=[experiment_name],
                         filter_string=f'tags.mlflow.runName = "{best_trial_name}"',
                         order_by=['attributes.start_time'])['run_id'])[-1]
    logging.info('Building Best model tags')
    model_tags = {
                  'name': experiment_name,
                  'run_id': best_trial_id,
                  'run_name': best_trial_name,
                  'metric': mape,
                  'feature_store_params': md,
                  }
    prev_model = None
    try:
        logging.info('Checking to see if model exists in registry')
        prev_model = client.get_model_version_by_alias(experiment_name, 'Staging')
        logging.info('Model Found')
    except:
        logging.info(f'No model with name {experiment_name} present! Creating new model.')
    if not prev_model:
        if mape < 0.15:
            logging.info('Current MAPE < 0.15. Saving to Registry with Version -> 1 and tag -> STAGING')
            mlflow.register_model(
                f'runs:/{best_trial_id}/{artifact_path}',
                experiment_name, tags=model_tags
            )
            client.set_registered_model_alias(experiment_name, 'Staging', '1')
            logging.info('Saving Model')
            mlflow.lightgbm.save_model(model, f'./mlruns/{experiment_id}/{best_trial_id}/artifacts/models/')
    else:
        logging.info('Comparing current model with previous best model.')
        prev_best_model_version = client.get_model_version_by_alias(experiment_name, 'Staging')
        prev_run_id = prev_best_model_version.run_id
        prev_run_version = int(prev_best_model_version.version)
        logging.info(f'Loading Previous model from Run ID -> {prev_run_id} and Run Version -> {prev_run_version}')
        prev_best_model = mlflow.pyfunc.load_model(f'runs:/{prev_run_id}/models')
        logging.info('Evaluating Previous best model with current test set')
        prev_model_preds = prev_best_model.predict(x_test)
        mape_old_model = mean_absolute_percentage_error(y_test, y_pred)
        logging.info(f'MAPE with previous best model -> {mape_old_model}')
        if mape >= mape_old_model:
            new_version = str(prev_run_version + 1)
            logging.info('New model performing better. Updating previous model alias from STAGING -> Archived')
            client.delete_registered_model_alias(experiment_name, "Staging")
            client.set_registered_model_alias(experiment_name, alias= 'Archived', version= str(prev_run_version))
            logging.info(f'Registering new model with version -> {best_trial_id} and version -> {new_version}')
            mlflow.register_model(
                f'runs:/{best_trial_id}/{artifact_path}',
                experiment_name, tags=model_tags
            )
            client.set_registered_model_alias(experiment_name, alias='Staging', version = new_version)
            logging.info('Saving New model')
            mlflow.lightgbm.save_model(model, f'./mlruns/{experiment_id}/{best_trial_id}/artifacts/models/')
