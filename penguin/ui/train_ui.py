"""
This script creates a page for training
"""

# Internal libraries
from os.path import abspath, dirname
import io
import uuid


# External libraries
import streamlit as st
import numpy as np
import pandas as pd
import mlflow

# Sagemaker and AWS libraries
import boto3
from sagemaker import LocalSession

# Internal libraries
from penguin.ui.train_ui_helpers import do_train
from penguin.bokeh_plots.scatterplot import plot_scatter
from penguin.class_utils.conf_mat_helpers import plot_confusion_matrix

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Some constants
IMAGE_NAME = 'penguin-xgb-training'
REGION_NAME = 'us-west-2'
ROLE = 'arn:aws:iam::784420883498:role/service-role/AmazonSageMaker-ExecutionRole-20200313T094543'
BUCKET_NAME = 'ky-blogs'
MODEL_SAVE_OBJ = 'penguins/'
MODEL_SAVE_PATH = 's3://' + BUCKET_NAME + '/' + MODEL_SAVE_OBJ

# Create an S3 client
s3 = boto3.resource('s3')

# Change working directory to 'test' directory
project_dir = dirname(dirname(dirname(abspath(__file__))))
print(project_dir)

@st.cache
def create_mlflow_exp():
    # Create a new experiment with unique ID
    exp_uniq_name = 'exp_' + uuid.uuid4().hex[:6].upper()
    mlflow.set_experiment(exp_uniq_name)

    return exp_uniq_name


def page(state):

    # Streamlit session state
    print(state.mlflow_res.shape)

    # Delete any existing mlflow experiments naned '0'
    try:
        mlflow.delete_experiment('0')
    except:
        pass

    # Create a new experiment with unique ID
    exp_uniq_name = create_mlflow_exp()

    # Title and Description
    st.title('Training')
    st.markdown("""
        This page guides you through the selection of algorithms and hyperparameters to tune the algorithms

        **Start by loading the data**.
        """)

    data_file = st.file_uploader("Upload data...", type="csv", key='train_data')
    try:
        text_io = io.TextIOWrapper(data_file)
    except:
        pass
    if data_file is not None:
        W = pd.read_csv(data_file)
        y = W.iloc[:,0]
        X = W.iloc[:,1:]
        st.write(W)

    # Options to select Model
    st.subheader('Select Training Method')
    model_types = ['XGBoost']
    sel_model = st.radio('Which model to use for training?', model_types)

    params = {}
    if sel_model=='XGBoost':
        # Get parameters for XGBoost
        params['num-rounds'] = st.sidebar.number_input(label="Number of boosting rounds or trees", min_value=5, max_value=100, value=10,
                                              step=1)
        params['max-depth'] = st.sidebar.number_input(label="Maximum depth of trees", min_value=2, max_value=10, value=2, step=1)
        params['eta'] = st.sidebar.number_input(label="Step size shrinkage for each boosting step", min_value=0.05, max_value=0.5, value=0.1,
                                              step=0.05)

        params['subsample'] = st.sidebar.number_input(label="Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting", min_value=0.0,
                                              max_value=1.0, value=0.75,
                                              step=0.05)
        params['colsample-bytree'] = st.sidebar.number_input(
            label="Subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed",
            min_value=0.0,
            max_value=1.0, value=0.75,
            step=0.05)
        params['num-folds'] = st.sidebar.number_input(label="Number of folds to use in cross-validation", min_value=2,
                                              max_value=20, value=10,
                                              step=1)

    # Sagemaker Training options
    instance_types = ['local', 'ml.m5.large', 'ml.m5.xlarge', 'ml.m5.4xlarge', 'ml.m5.24xlarge', 'ml.g4dn.xlarge', 'ml.g4dn.4xlarge', 'ml.g4dn.16xlarge']
    sagemaker_instance = st.sidebar.selectbox('Instance type for Sagemaker training', instance_types)

    if len(params) > 0:
        cv_submit = st.sidebar.button('Run CV')
        train_submit = st.sidebar.button('Run Training')
        best_train_submit = st.sidebar.button('Run Training with Best CV Parameters')
        print(state.best_train_submit_button)
        print(cv_submit)
        if best_train_submit and state.mlflow_res.shape[0]==0:
            st.warning('Please run at least one CV run before training on the best CV parameters')
        if cv_submit or train_submit or best_train_submit:
            if sel_model == 'XGBoost':
                model_title = 'XGBoost'
                image_name = IMAGE_NAME
                model_name_suffix = 'penguin_xgb_model.json'

            if sagemaker_instance == 'local':
                sagemaker_session = LocalSession(boto_session=boto3.Session(region_name=REGION_NAME))
                instance_type = 'local'
                params['categorical-columns'] = "island,sex"
                params['train-file'] = "penguins.csv"
                if cv_submit:
                    params['do-cv'] = 1
                else:
                    params['do-cv'] = 0

                if best_train_submit:
                    state.best_train_submit_button = True
                    param_names = [col.split('params.')[1] for col in state.mlflow_res.columns if col.startswith('params.')]
                    for key in param_names:
                        params[key] = state.mlflow_res.loc[state.mlflow_res.index[state.sel_best_run]-1, 'params.'+ key]
                    print(1)

                train_res = do_train(W, project_dir, sagemaker_session, instance_type, role=ROLE, image_name=IMAGE_NAME, params=params, model_name_suffix=model_name_suffix)


            # Select mlflow runs
            mlflow_res = mlflow.search_runs()
            logger.info(mlflow_res.columns)
            sel_cols = ['run_id', 'experiment_id']
            sel_cols.extend([col for col in mlflow_res.columns if col.startswith(('metrics', 'params'))])
            logger.info(sel_cols)
            mlflow_res = mlflow_res.loc[:, sel_cols]
            mlflow_res = mlflow_res.set_index('run_id')
            mlflow_res.index = np.arange(1, len(mlflow_res) + 1)

            #Assign the mlflow_res variable to the state. This will persisit for the rest of the session unless modified again here
            state.mlflow_res = mlflow_res
            print(state.mlflow_res)

    if state.mlflow_res.shape[0] != 0:
        st.markdown("**Metrics and Parameters Table**")
        st.write(state.mlflow_res)
        sel_best = st.selectbox(
            'Which parameters to choose for the model? Pick a row number from the table above',
            state.mlflow_res.index)
        state.sel_best_run = sel_best

        # Plot CV plots
        sel_x = st.selectbox('X-axis for the CV plot', [col for col in state.mlflow_res.columns if col.startswith('params')])
        sel_y = st.selectbox('X-axis for the CV plot', [col for col in state.mlflow_res.columns if col.startswith('metrics')])
        print(sel_x, sel_y)
        plot_scatter(state.mlflow_res, sel_x, sel_y)

    if state.best_train_submit_button:
        # Plot Confusion Matrix
        plot_confusion_matrix(train_res[1].get_label().astype(int), train_res[5],
                              classes=np.asarray(list(train_res[2].keys())),
                              title='Confusion matrix, without normalization')
        plot_confusion_matrix(train_res[1].get_label().astype(int), train_res[5],
                              classes=np.asarray(list(train_res[2].keys())), normalize=True,
                              title='Normalized confusion matrix')

