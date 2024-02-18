An end to end solution to forecast the daily number of San Francisco BART riders.

Tools Used:

- Data Pipeline orchestration: Airflow
- Feature Store: Hopsworks
- ML Model Training and Registry: MLFlow
- Model: LightGBM
- Storage: Minio
- UI: Streamlit

Data Sources:
- Historical BART data - [https://www.sf.gov/data/san-francisco-bart-ridership]()
- NFL home game data - [https://pypi.org/project/nfl-data-py]()
- Weather Data - [https://archive-api.open-meteo.com/v1/archive]()
- Holiday Data - [https://pypi.org/project/holidays]()

<p>
    <b>***NOTES: Right now the workflow is set for the data pipeline, model training pipeline and the serving pipeline. Some of the credential info is saved within the AirflowUI itself or in the .py files. THIS IS NOT BEST PRACTICE, however it is a placeholder for now. More info on this below.
I will be incorporating the monitoring and retraining pipelines soon.

I will also be working on containerizing everything sometime in the near future.***
    </b>
</p>

Data Pipeline steps:
- Set up Airflow by following the steps:
  - Create your python virtual environment. The python version I used is 3.10.11
  - Inside your project environment, create a folder called /dag. All your data pipeline code will go inside this folder
  - Set AIRFLOW_HOME to the project folder. For me this was "export AIRFLOW_HOME = Users/shawar/project_folder".
  - I used airflow 2.8.1 for this. You can set the variable name 'AIRFLOW_VERSION=2.8.1'
  - Set python version -> PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
  - Set constraint variable -> CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
  - Finally, you can pip install airflow -> pip install "apache-airflow[async,postgres,google]==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
  - Once the installation is done, migrate the db -> airflow db migrate
  - Then create the user credentials -> airflow users create \
                                        --username [username]
                                        --firstname [first name]
                                        --lastname [last name]
                                        --role Admin
                                        email [email]
  - Then you can start the airflow server -> airflow webserver. This automatically starts the webserver on localhost:8080.
  - Then start the airflow scheduler -> airflow scheduler
  - Install all the necessary packages -> pip install -r requirements.txt
  - One of the packages is Minio. I used this to store data. You'll need to run the minio server by -> minio server 'path/to/dag/folder'
  - The feature store I used to store the processed data is Hopsworks. You'll need to create an account and store the API key to use it. Follow the instructions here -> [https://docs.hopsworks.ai/3.5/user_guides/projects/auth/registration]()
  - You'll also need an APP token to download the BART data. Follow the instructions here -> [https://data.sfgov.org/login]()
  - <b>**NOTE: Most of the credential info I've needed, I've either placed in the dags/data_creation/feature_store_variables.py file, or stored as a variable in Airflow itelf. THIS IS NOT GOOD PRACTICE. I've done it this way for now as a placeholder and will replace it soon. You can change this too by either using a secret manager or a .dot.env file ([https://configu.com/blog/dotenv-managing-environment-variables-in-node-python-php-and-more]())**</b>
  - The data I've stored in Airflow itself are: BART App token, BART data URL, the BART stop for which I trained the model.
  - The other credentials I've stored in the dags/data_creation/feature_store_variables.py are the hopsworks API key and the names for the hopsworks feature groups, feature views etc.
  - Your minio credentials will go into the variables.py file.
  - To set the required variables on Airflow, go to the web browser and open 127.0.0.1:8080 to view the Airflow console.
  - Click on the Admin tab and select Variables, then click on the + button and enter a key, value and an optional description for your Airflow variable.
  - The key value pairs are  {'url': 'data.sfgov.org', 'sfgov_token': your sf gov app token, 'stop': stop name you want to run the model for}
  - To understand more about the BART stop, follow the link -> [https://data.sfgov.org/Transportation/BART-Daily-Station-Exits/m2xz-p7ja/about_data]()
  - Once you're ready, run the data pipeline with python path/to/data_pipeline.py. Then go to the UI, click on DAGS, click on the 'Active' tab. Click Play.
  - Once the pipeline runs, you should see the data in the Hopsworks feature store.

Model training pipeline:
- Set up the training pipeline with the following steps:
  - Open a terminal and enter the command -> mlflow ui.
  - You can view the UI by going to the link 127.0.0.1:5000.
  - I use Optuna to tune the hyperparameters. You can make edits to the parameters in the params variable in the objective function in build_model.py. For best practices, you can move that variable to the model_variables.py file.
  - You can also change how many months you want to use for validation data in the num_months_var variable in model_variables.py.
  - <b>**NOTE: I've left a few placeholders as I wanted to also build a linear regression model as a baseline model. I will be making edits to that soon, but in the meantime you can try to do it as a personal exercise too.**</b>
  - Once you're ready to run the training pipeline, just run python path/to/build_model.py.
  - This will create a 'lightgbm-experiment' experiment in the UI a nested run, each sub run associated with a set of hyperparameters.
  - Once the pipeline is run, if the performance is good enough, it will store a model in the model artifacts with a 'Serving' tag.
  - <b>**NOTE: I will be updating the workflow in the new future to accommodate for monitoring and retraining, so stay tuned for that.**</b>

Model prediction pipeline:
- Set up the model serving pipeline with the following steps:
  - The model serving pipeline is a little different in the sense that I use streamlit to create a UI.
  - The UI requires you to choose a date upto which you want to make the predictions for. The weather API I use has a limit of giving the weather forecast for 16 days, so the UI wont allow you to choose a date beyond 16 days.
  - To run the predictions, run python -m streamlit run path/to/predict.py. This will open the UI. Click on the date you want, and click on 'Estimate Passengers'. This will run the serving pipeline, and generate the line graph showing you some of the historical data as well as the forecast data in an interactable plot.
