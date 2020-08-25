"""
This script creates a page for training
"""

# Internal libraries
import os
import tempfile

# External libraries
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

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def do_train(df, project_dir, sagemaker_session, instance_type, role, image_name, params, model_name_suffix):

    if instance_type=='local':
        with tempfile.TemporaryDirectory(dir=os.path.join(project_dir, 'sagemaker_tmp/out_tar')) as tmpdir:
            df.to_csv(os.path.join(tmpdir, 'penguins.csv'), index=False)
            model_data_path = 'file://' + os.path.join(tmpdir, 'penguins.csv')
            model_save_path = 'file://' + tmpdir

            estimator = Estimator(
                role=role,
                sagemaker_session=sagemaker_session,
                train_instance_count=1,
                train_instance_type=instance_type,
                image_name=image_name,
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
            if os.path.exists(os.path.join(tmpdir, 'data', 'dmatrix_train.data')):
                dtrain = xgb.DMatrix(os.path.join(tmpdir, 'data', 'dmatrix_train.data'))
            if os.path.exists(os.path.join(tmpdir, 'data', 'dmatrix_test.data')):
                dtest = xgb.DMatrix(os.path.join(tmpdir, 'data', 'dmatrix_test.data'))
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
                return (dtrain, dtest, xgb_loaded, train_preds, test_preds, all_metrics)

            else:
                print('Model could not be found in the tar.gz file')

    else:
        sagemaker_session =Session(boto_session=boto3.Session(region_name=REGION_NAME))
        instance_type = sagemaker_instance
        image_name = '784420883498.dkr.ecr.us-west-1.amazonaws.com/' + image_name + ':latest'

        # with tempfile.TemporaryDirectory(dir=os.path.join(project_dir, 'sagemaker_tmp/out_tar')) as tmpdir:
        #     W.to_csv(os.path.join(tmpdir, 'genopheno.csv'), index=False)
        #
        #     s3_tmpdir = unique_name_from_base('sagemaker-coa-tmp')
        #     model_save_path = MODEL_SAVE_PATH + s3_tmpdir
        #     params['train-file'] = 'genopheno.csv'
        #     estimator = Estimator(
        #         role=ROLE,
        #         sagemaker_session=sagemaker_session,
        #         train_instance_count=1,
        #         train_instance_type=instance_type,
        #         image_name=image_name,
        #         output_path=model_save_path,
        #         hyperparameters=params)
        #
        #     # Send training data to s3
        #     model_data_path = estimator.sagemaker_session.upload_data(
        #         path=os.path.join(tmpdir, 'genopheno.csv'), bucket=BUCKET_NAME,
        #         key_prefix='streamlit/' + s3_tmpdir)
        #
        #     st.button('Cancel')
        #     with st.spinner('Training...'):
        #         estimator.fit(model_data_path, job_name=s3_tmpdir, wait=True)  # Not sure if it would work with relative paths
        #     st.balloons()
        #
        #     # Download the output files
        #     out_file = estimator.model_data.replace("model.tar.gz", "output.tar.gz")
        #     obj_name = os.path.relpath(out_file, 's3://' + BUCKET_NAME)
        #     s3.Bucket(BUCKET_NAME).download_file(obj_name, os.path.join(tmpdir, 'output.tar.gz'))
        #
        #     # Extract the cross validation predictions
        #     tf = tarfile.open(os.path.join(tmpdir, 'output.tar.gz'))
        #     tf.extractall(tmpdir)
        #     if os.path.exists(os.path.join(tmpdir, 'ycv.csv')):
        #         ycv = np.loadtxt(os.path.join(tmpdir, 'ycv.csv'))
        #
        #     # Download the model files
        #     obj_name = os.path.relpath(estimator.model_data, 's3://' + BUCKET_NAME)
        #     s3.Bucket(BUCKET_NAME).download_file(obj_name, os.path.join(tmpdir, 'model.tar.gz'))
        #
        #     # Extract the model
        #     tf = tarfile.open(os.path.join(tmpdir, 'model.tar.gz'))
        #     tf.extractall(tmpdir)
        #     if os.path.exists(os.path.join(tmpdir, model_name_suffix)):
        #         with open(os.path.join(tmpdir, model_name_suffix), 'rb') as pickle_file:
        #             model = pickle.load(pickle_file)
        #     else:
        #         print('Model could not be found in the tar.gz file')
        #
        #     # Clean up s3_tmpdir
        #     s3_client = boto3.client('s3')
        #     for ct in s3_client.list_objects_v2(Bucket='learn-coa', Prefix='streamlit/' + s3_tmpdir + '/')['Contents']:
        #         s3.Object(BUCKET_NAME, ct['Key']).delete()

