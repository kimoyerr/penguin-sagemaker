#!/usr/bin/python
#
# Inspired From: https://github.com/aws/sagemaker-inference-toolkit/blob/master/test/integration/local/test_dummy_multi_model.py
# Some Parts From: https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/test/integration/local/test_container.py
# Some Other Parts From: https://github.com/aws/sagemaker-tensorflow-training-toolkit/blob/master/test/integration/local/test_training.py

import json
import os
from os.path import abspath, dirname
import subprocess
import sys
import time
import tarfile
import shutil

# External libraries
import pytest
import pickle
import xgboost as xgb

# Sagemaker and AWS libraries
import boto3
from sagemaker import LocalSession, Session
from sagemaker.estimator import Estimator

# Logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Some constants
IMAGE_NAME = 'penguin-xgb-training'
REGION_NAME = 'us-west-2'

# Change working directory to 'test' directory
cwd_dir = dirname(dirname(dirname(abspath(__file__))))
os.chdir(cwd_dir)
print(os.getcwd())
test_dir = dirname(dirname(dirname(abspath(__file__))))
project_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))  # The path to the parent test directory


@pytest.fixture(scope='session', name='build_xgb_image', autouse=True)
def fixture_build_xgb_image():
    build_image = IMAGE_NAME
    command = "sh " + project_dir + '/scripts/build_docker_image_penguin_training.sh xgb ' + project_dir
    logger.debug(command)
    proc = subprocess.check_call(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

    return proc


@pytest.fixture(scope='session')
def sagemaker_local_session():
    return LocalSession(boto_session=boto3.Session(region_name=REGION_NAME))


@pytest.fixture(scope='session')
def account_id(request):
    return request.config.getoption('--account-id')


@pytest.fixture
def instance_type(request, processor):
    provided_instance_type = request.config.getoption('--instance-type')
    default_instance_type = 'ml.c4.xlarge' if processor == 'cpu' else 'ml.p2.xlarge'
    return provided_instance_type if provided_instance_type is not None else default_instance_type


###################################################################################################################

def test_xgb_train_container_cpu(sagemaker_local_session, build_xgb_image):
    model_save_path = 'file:///home/ubuntu/penguin-sagemaker/test/resources/models_tar'  # Has to be absolute path for local
    if os.path.exists(os.path.join(test_dir, 'resources/models_tar', 'model.tar.gz')):
        os.remove(os.path.join(test_dir, 'resources/models_tar', 'model.tar.gz'))
        time.sleep(3)
    model_data_path = 'file://' + os.path.join(test_dir, 'resources/data/')

    estimator = Estimator(
        role='arn:aws:iam::784420883498:role/service-role/AmazonSageMaker-ExecutionRole-20200313T094543',
        sagemaker_session=sagemaker_local_session,
        train_instance_count=1,
        train_instance_type='local',
        image_name=IMAGE_NAME,
        output_path=model_save_path,
        hyperparameters={"train-file": "penguins.csv",
                         "max-depth": 3,
                         "categorical-columns": 'island,sex'})

    estimator.fit(model_data_path, wait=True)  # Not sure if it would work with relative paths

    _assert_files_exist_in_tar(model_save_path, ['penguin_xgb_model.json'])


def test_xgb_save_and_load(X, xgb_matrix):
    model_save_path = 'file://' + os.path.join(test_dir, 'resources/models_tar',
                                               'model.tar.gz')  # Has to be absolute path for local
    model_save_path = model_save_path.replace('file://', '')

    # Copy the file to a new file name
    new_model_save_path = model_save_path.replace('model.tar.gz', 'xgb_model.tar.gz')
    shutil.copy(model_save_path, new_model_save_path)

    # Extract the tar.gz file
    tf = tarfile.open(model_save_path)
    tf.extractall(os.path.join(test_dir, 'resources/models_tar'))
    model_file = model_save_path.replace('model.tar.gz', 'penguin_xgb_model.json')
    # Load model
    xgb_loaded = xgb.Booster()
    xgb_loaded.load_model(model_file)

    # Also save the model files in the saved_models folder
    if not os.path.exists('resources/saved_models/xgb/'):
        os.mkdir('resources/saved_models/xgb/')
    tf.extractall(os.path.join(test_dir, 'resources/saved_models/xgb'))

    assert (xgb_loaded.predict(xgb_matrix['train']).shape[0] + xgb_loaded.predict(xgb_matrix['test']).shape[0]) == \
           X.shape[0]
    assert 'penguin_xgb_model.json' in os.listdir(os.path.join(test_dir, 'resources/saved_models/xgb'))


def _assert_files_exist_in_tar(output_path, files):
    if output_path.startswith('file://'):
        output_path = output_path[7:]
    model_file = os.path.join(output_path, 'model.tar.gz')
    with tarfile.open(model_file) as tar:
        for f in files:
            tar.getmember(f)
