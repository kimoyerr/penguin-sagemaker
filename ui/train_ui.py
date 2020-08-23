"""
This script creates a page for training
"""

# Internal libraries
import os
from os.path import abspath, dirname
import tempfile
import io
import uuid


# External libraries
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import tarfile
import xgboost as xgb
import mlflow

# Sagemaker and AWS libraries
import boto3
from sagemaker import LocalSession, Session
from sagemaker.estimator import Estimator
from sagemaker.utils import unique_name_from_base

# Internal libraries
from penguin.penguin_xgb import dummify_X, encode_y, create_xgb_matrix, fit_xgb

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
project_dir = dirname(dirname(abspath(__file__)))
print(project_dir)

@st.cache
def create_mlflow_exp():
    # Create a new experiment with unique ID
    exp_uniq_name = 'exp_' + uuid.uuid4().hex[:6].upper()
    mlflow.set_experiment(exp_uniq_name)

    return exp_uniq_name


def page():
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
        submit = st.sidebar.button('Train Model')
        if submit:
            if sel_model == 'XGBoost':
                model_title = 'XGBoost'
                image_name = IMAGE_NAME
                model_name_suffix = 'penguin_xgb_model.json'

            if sagemaker_instance == 'local':
                sagemaker_session = LocalSession(boto_session=boto3.Session(region_name=REGION_NAME))
                instance_type = 'local'

                with tempfile.TemporaryDirectory(dir=os.path.join(project_dir, 'sagemaker_tmp/out_tar')) as tmpdir:
                    W.to_csv(os.path.join(tmpdir, 'penguins.csv'), index=False)
                    model_data_path = 'file://' + os.path.join(tmpdir, 'penguins.csv')
                    model_save_path = 'file://' + tmpdir

                    params['categorical-columns'] = "island,sex"
                    params['train-file'] = "penguins.csv"
                    params['do-cv'] = 1
                    estimator = Estimator(
                        role=ROLE,
                        sagemaker_session=sagemaker_session,
                        train_instance_count=1,
                        train_instance_type=instance_type,
                        image_name=IMAGE_NAME,
                        output_path=model_save_path,
                        hyperparameters=params)

                    st.button('Cancel')
                    with st.spinner('Training...'):
                        estimator.fit(model_data_path, wait=True) # Not sure if it would work with relative paths
                    st.balloons()

                    # Extract the model
                    logger.info(os.listdir(tmpdir))
                    tf = tarfile.open(os.path.join(tmpdir, 'model.tar.gz'))
                    tf.extractall(tmpdir)

                    # Extract the cross validation results
                    tf = tarfile.open(os.path.join(tmpdir, 'output.tar.gz'))
                    tf.extractall(tmpdir)
                    logger.info(os.listdir(os.path.join(tmpdir, 'data')))
                    if os.path.exists(os.path.join(tmpdir, 'data', 'train_preds.csv')):
                        train_preds = np.loadtxt(os.path.join(tmpdir, 'data', 'train_preds.csv'))
                    if os.path.exists(os.path.join(tmpdir, 'data', 'test_preds.csv')):
                        test_preds = np.loadtxt(os.path.join(tmpdir, 'data', 'test_preds.csv'))
                    if os.path.exists(os.path.join(tmpdir, 'data', 'all_metrics.csv')):
                        all_metrics = np.loadtxt(os.path.join(tmpdir, 'data', 'all_metrics.csv'))
                    if os.path.exists(os.path.join(tmpdir, 'data', 'penguin_xgb_cv.csv')):
                        cv_res = pd.read_csv(os.path.join(tmpdir, 'data', 'penguin_xgb_cv.csv'))
                        logger.info(cv_res)
                        # Track cv metrics using mlflow
                        with mlflow.start_run() as run:
                            mlflow.log_param("max-depth", params["max-depth"])
                            mlflow.log_param("eta", params["eta"])
                            mlflow.log_param("subsample", params["subsample"])
                            mlflow.log_param("colsample-bytree", params["colsample-bytree"])
                            # Log a metric; metrics can be updated throughout the run
                            mlflow.log_metric("train-merror-mean", cv_res.tail(1).loc[:, 'train-merror-mean'].squeeze())
                            mlflow.log_metric("test-merror-mean", cv_res.tail(1).loc[:, 'test-merror-mean'].squeeze())

                    if os.path.exists(os.path.join(tmpdir, model_name_suffix)):
                        model_file = os.path.join(tmpdir, 'penguin_xgb_model.json')
                        # Load model
                        xgb_loaded = xgb.Booster()
                        xgb_loaded.load_model(model_file)

                        logger.info(os.getcwd())

                    else:
                        print('Model could not be found in the tar.gz file')

            else:
                sagemaker_session =Session(boto_session=boto3.Session(region_name=REGION_NAME))
                instance_type = sagemaker_instance
                image_name = '784420883498.dkr.ecr.us-west-1.amazonaws.com/' + image_name + ':latest'

                with tempfile.TemporaryDirectory(dir=os.path.join(project_dir, 'sagemaker_tmp/out_tar')) as tmpdir:
                    W.to_csv(os.path.join(tmpdir, 'genopheno.csv'), index=False)

                    s3_tmpdir = unique_name_from_base('sagemaker-coa-tmp')
                    model_save_path = MODEL_SAVE_PATH + s3_tmpdir
                    params['train-file'] = 'genopheno.csv'
                    estimator = Estimator(
                        role=ROLE,
                        sagemaker_session=sagemaker_session,
                        train_instance_count=1,
                        train_instance_type=instance_type,
                        image_name=image_name,
                        output_path=model_save_path,
                        hyperparameters=params)

                    # Send training data to s3
                    model_data_path = estimator.sagemaker_session.upload_data(
                        path=os.path.join(tmpdir, 'genopheno.csv'), bucket=BUCKET_NAME,
                        key_prefix='streamlit/' + s3_tmpdir)

                    st.button('Cancel')
                    with st.spinner('Training...'):
                        estimator.fit(model_data_path, job_name=s3_tmpdir, wait=True)  # Not sure if it would work with relative paths
                    st.balloons()

                    # Download the output files
                    out_file = estimator.model_data.replace("model.tar.gz", "output.tar.gz")
                    obj_name = os.path.relpath(out_file, 's3://' + BUCKET_NAME)
                    s3.Bucket(BUCKET_NAME).download_file(obj_name, os.path.join(tmpdir, 'output.tar.gz'))

                    # Extract the cross validation predictions
                    tf = tarfile.open(os.path.join(tmpdir, 'output.tar.gz'))
                    tf.extractall(tmpdir)
                    if os.path.exists(os.path.join(tmpdir, 'ycv.csv')):
                        ycv = np.loadtxt(os.path.join(tmpdir, 'ycv.csv'))

                    # Download the model files
                    obj_name = os.path.relpath(estimator.model_data, 's3://' + BUCKET_NAME)
                    s3.Bucket(BUCKET_NAME).download_file(obj_name, os.path.join(tmpdir, 'model.tar.gz'))

                    # Extract the model
                    tf = tarfile.open(os.path.join(tmpdir, 'model.tar.gz'))
                    tf.extractall(tmpdir)
                    if os.path.exists(os.path.join(tmpdir, model_name_suffix)):
                        with open(os.path.join(tmpdir, model_name_suffix), 'rb') as pickle_file:
                            model = pickle.load(pickle_file)
                    else:
                        print('Model could not be found in the tar.gz file')

                    # Clean up s3_tmpdir
                    s3_client = boto3.client('s3')
                    for ct in s3_client.list_objects_v2(Bucket='learn-coa', Prefix='streamlit/' + s3_tmpdir + '/')['Contents']:
                        s3.Object(BUCKET_NAME, ct['Key']).delete()

            # Select mlflow runs
            mlflow_res = mlflow.search_runs()
            logger.info(mlflow_res.columns)
            sel_cols = ['run_id', 'experiment_id']
            sel_cols.extend([col for col in mlflow_res.columns if col.startswith(('metrics', 'params'))])
            logger.info(sel_cols)
            mlflow_res = mlflow_res.loc[:, sel_cols]
            mlflow_res = mlflow_res.set_index('run_id')
            mlflow_res.index = np.arange(1, len(mlflow_res) + 1)
            st.write(mlflow_res)
            sel_best = st.selectbox('Which parameters to choose?Pick a row number from the table above', mlflow_res.index)

            # # Create the XGBMatrix from the input data
            # # Pre-process data
            # y = W.iloc[:, 1]
            # X = W.iloc[:, 2:]
            # # Convert categorical columns to one-hot encodings
            # cat_columns = ["island","sex"]
            # dummy_X = dummify_X(X, cat_columns)
            # # Convert the categorical labels in the target to integer labels
            # label_encoder = encode_y(y)
            # encoded_y = label_encoder.transform(y)
            #
            # # Create the xgboost matrices for training and evaluation
            # xgb_matrix = create_xgb_matrix(dummy_X, y, label_encoder)
            # print(xgb_matrix.keys())
            #
            # # Calculate metrics and plot
            # rho = np.corrcoef(y, ycv)[0, 1] ** 2
            # plt.figure()
            # plt.scatter(y, ycv, s=2)
            # lim = plt.gca().get_xlim()
            # plt.plot(lim, lim)
            # plt.ylim(lim)
            # plt.title(model_title)
            # plt.xlabel('Measured')
            # plt.ylabel('Predicted')
            # plt.text(0.9*lim[0] + 0.1*lim[1], 0.1*lim[0] + 0.9*lim[1], '$R^2$=' + '{:.2f}'.format(rho))
            # st.pyplot()
            #
            # session.set('model', model)
            # st.subheader('Trained model')
            #
            # # Load the pickled file and let it be downloadable
            # st.markdown(get_object_download_link(model, 'model.coa', 'Download model file'), unsafe_allow_html=True)
