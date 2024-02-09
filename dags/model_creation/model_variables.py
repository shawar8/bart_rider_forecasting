import lightgbm as lgb


tag_name = 'bart-forecasting'
experiment_name = 'lightgbm-bart'
tracking_uri = "http://127.0.0.1:5000"
num_months_val = 3
description_dict = {'lightgbm-bart': 'This is the LightGBM model to predict BART daily ridership.',
                    'linreg-bart': 'This is the Linear Regression model to predict BART daily ridership.'}

experiment_tags = {'project_name': 'bart-forecasting',
                   'mlflow.note.content': description_dict[experiment_name]}

experiment_dict = {'lightgbm-bart': {
                'description': 'This is the LightGBM model to predict BART daily ridership.',
                'artifact_path': 'lightgbm_artifact',
                },
                'linreg-bart': {
                'description': 'This is the linear regression model to predict BART daily ridership.',
                'artifact_path': 'linreg_artifact',
                }}
artifact_path = experiment_dict[experiment_name]['artifact_path']
model_dict = {'lightgbm-bart': lgb.LGBMRegressor(),
              'linreg-bart': 'Linear Regression model'}
hyperopt_dict = {'lightgbm-bart': 'lightgbm_opt_bart',
                 'linreg-bart': 'linreg_opt_bart'}
