#!/usr/bin/python
"""
# Inspired From: https://github.com/aws/sagemaker-inference-toolkit/blob/master/test/integration/local/test_dummy_multi_model.py
# Some Parts From: https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/test/integration/local/test_container.py
# Some Other Parts From: https://github.com/aws/sagemaker-tensorflow-training-toolkit/blob/master/test/integration/local/test_training.py
"""


import json
import os
from os.path import abspath, dirname
import subprocess
import sys
import time
import tarfile
import shutil
import pickle

# External libraries
import pytest
import xgboost as xgb

# Sagemaker and AWS libraries
import boto3
from sagemaker import LocalSession, Session
from sagemaker.estimator import Estimator
from sagemaker.utils import unique_name_from_base
from six.moves.urllib.parse import urlparse

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Some constants
XGB_IMAGE_NAME = '577756101237.dkr.ecr.us-west-2.amazonaws.com/penguin-xgb-training:latest'
REGION_NAME = 'us-west-2'
ROLE = 'arn:aws:iam::577756101237:role/Sagemaker_execution'
BUCKET_NAME = 'ky-blogs'
MODEL_SAVE_OBJ = 'penguins/tmp/models'
MODEL_SAVE_PATH = 's3://' + BUCKET_NAME + '/' + MODEL_SAVE_OBJ


# Change working directory to 'test' directory
cwd_dir = dirname(dirname(dirname(abspath(__file__))))
os.chdir(cwd_dir)
logger.debug(os.getcwd())
test_dir = dirname(dirname(dirname(abspath(__file__))))
project_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))# The path to the parent test directory

# Create an S3 client
s3 = boto3.resource('s3')

@pytest.fixture(scope='session', name='build_xgb_image', autouse=True)
def fixture_build_xgb_image():
    build_image = XGB_IMAGE_NAME
    command = "sh " + project_dir + '/scripts/build_docker_image_penguin_training.sh xgb ' + project_dir
    logger.debug(command)
    proc = subprocess.check_call(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

    return proc

@pytest.fixture(scope='session')
def processor():
    return 'cpu'

@pytest.fixture(scope='session')
def sagemaker_session():
    return Session(boto_session=boto3.Session(region_name=REGION_NAME))

@pytest.fixture(scope='session')
def account_id(request):
    return request.config.getoption('--account-id')

@pytest.fixture(scope='session')
def instance_type(request, processor):
    default_instance_type = 'ml.m5.large' if processor == 'cpu' else 'ml.g4dn.xlarge'
    return default_instance_type


def test_xgb_train_container_cpu(sagemaker_session, instance_type):
    training_data_path = os.path.join(test_dir, 'resources/data/')
    estimator = Estimator(
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=1,
        train_instance_type=instance_type,
        image_name=XGB_IMAGE_NAME,
        output_path=MODEL_SAVE_PATH,
        hyperparameters={"train-file": "penguins.csv",
                         "max-depth": 3,
                         "categorical-columns": 'island,sex'})

    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(training_data_path, 'penguins.csv'), bucket = BUCKET_NAME, key_prefix='penguins/tmp')
    estimator.fit(inputs, job_name=unique_name_from_base('test-sagemaker-xgb-training'))

    # Clean up the models folder and re-create it
    if os.path.exists(os.path.join(test_dir, 'resources/models_tar')):
        shutil.rmtree(os.path.join(test_dir, 'resources/models_tar'))
        os.mkdir(os.path.join(test_dir, 'resources/models_tar'))

    # Download the model files
    obj_name = os.path.relpath(estimator.model_data, 's3://' + BUCKET_NAME)
    s3.Bucket(BUCKET_NAME).download_file(
        obj_name, os.path.join(test_dir, 'resources/models_tar/model.tar.gz'))

    _assert_s3_file_exists(sagemaker_session.boto_region_name, estimator.model_data)

def test_gbr_save_and_load(X, xgb_matrix):
    model_save_path = 'file://' + os.path.join(test_dir, 'resources/models_tar', 'model.tar.gz') # Has to be absolute path for local
    model_save_path = model_save_path.replace('file://', '')

    # Extract the tar.gz file
    tf = tarfile.open(model_save_path)
    tf.extractall(os.path.join(test_dir, 'resources/models_tar'))
    model_file = model_save_path.replace('model.tar.gz', 'penguin_xgb_model.json')

    # Load model
    xgb_loaded = xgb.Booster()
    xgb_loaded.load_model(model_file)

    assert (xgb_loaded.predict(xgb_matrix['train']).shape[0] + xgb_loaded.predict(xgb_matrix['test']).shape[0]) == \
           X.shape[0]

def _assert_s3_file_exists(region, s3_url):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource('s3', region_name=region)
    s3.Object(parsed_url.netloc, parsed_url.path.lstrip('/')).load()