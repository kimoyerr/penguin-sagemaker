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


# External libraries
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score
import pytest
import xgboost as xgb
import requests

# Sagemaker and AWS libraries
import boto3
from sagemaker import LocalSession, Session
from sagemaker.estimator import Estimator

# Some constants
IMAGE_NAME = 'penguin-xgb-training'
REGION_NAME = 'us-west-2'

# Change working directory to 'test' directory
cwd_dir = dirname(dirname(abspath(__file__)))
os.chdir(cwd_dir)
print(os.getcwd())


# Create a docker container
@pytest.fixture(scope='session', autouse=True)
def container():

    try:
        # Remove any existing docker volumes that have the same name and also any running docker containers
        cmd = 'docker stop $(docker ps -aq)'
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)  # Wait to close all running containers
        cmd = 'docker rm $(docker ps -aq)'
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)  # Wait to close all running containers
        volume_out = subprocess.Popen('docker volume ls'.split(), stdout=subprocess.PIPE)
        if b'model_volume' in volume_out.communicate()[0]:
            cmd = 'docker stop $(docker ps -aq)'
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(p.communicate())
            print('here')
            subprocess.check_call('docker volume rm model_volume'.split())

        command = (
            "docker run --name penguin-xgb-train " + # Mind the trailing space
            IMAGE_NAME
        )

        print(command)
        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)
        yield proc.pid
    finally:
        subprocess.check_call("docker rm -f penguin-xgb-train".split())

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

def test_xgb_train_container_cpu(sagemaker_local_session):
    model_save_path = 'file:///home/ubuntu/penguin-sagemaker/test/resources/models_local_docker' # Has to be absolute path for local
    os.remove('/home/ubuntu/penguin-sagemaker/test/resources/models_local_docker/model.tar.gz')
    time.sleep(3)

    estimator = Estimator(
        role='arn:aws:iam::784420883498:role/service-role/AmazonSageMaker-ExecutionRole-20200313T094543',
        sagemaker_session=sagemaker_local_session,
        train_instance_count=1,
        train_instance_type='local',
        image_name=IMAGE_NAME,
        output_path=model_save_path,
        hyperparameters={"max-depth": 2,
                         "categorical-columns": 'island,sex'})

    estimator.fit("file:///home/ubuntu/penguin-sagemaker/test/resources/data/", wait=True) # Not sure if it would work with relative paths

    _assert_files_exist_in_tar(model_save_path, ['penguin_xgb_model.json'])


def _assert_files_exist_in_tar(output_path, files):
    if output_path.startswith('file://'):
        output_path = output_path[7:]
    model_file = os.path.join(output_path, 'model.tar.gz')
    with tarfile.open(model_file) as tar:
        for f in files:
            tar.getmember(f)