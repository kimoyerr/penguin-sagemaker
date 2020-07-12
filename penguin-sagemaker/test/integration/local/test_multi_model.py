#!/usr/bin/python
#
# Original From: https://github.com/aws/sagemaker-inference-toolkit/blob/master/test/integration/local/test_dummy_multi_model.py
# Some Parts From: https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/test/integration/local/test_container.py

import json
import os
import subprocess
import sys
import time

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

BASE_URL = "http://0.0.0.0:8080/"
PING_URL = BASE_URL + "ping"
INVOCATION_URL = BASE_URL + "models/{}/invoke"
MODELS_URL = BASE_URL + "models"
DELETE_MODEL_URL = BASE_URL + "models/{}"
IMAGE_NAME = "coa-inference:latest" # Make sure this image has already been built and is available on the local machine


# Fixtures
@pytest.fixture(scope="session", name="df")
def fixture_df():
    df = pd.read_csv('resources/data/penguins.csv', index_col=0)
    return df

@pytest.fixture(scope="session", name="X")
def fixture_X(df):
    X = df.iloc[:,1:]
    return X

@pytest.fixture(scope="session", name="y")
def fixture_y(df):
    y = df.iloc[:,0]
    return y

@pytest.fixture(scope="session", name="dummy_X")
def fixture_dummify_X(X):

    sel_columns = ['island', 'sex']
    dummy_X = X.copy()
    for col in sel_columns:
        dummy_X = pd.concat([dummy_X.drop(col, axis=1), pd.get_dummies(dummy_X[col])], axis=1)

    return dummy_X

@pytest.fixture(scope="session", name="encoded_y")
def fixture_encode_y(y):

    le = LabelEncoder()
    encoded_y = le.fit_transform(y)

    return encoded_y

@pytest.fixture(scope="session", name="xgb_matrix")
def fixture_xgb_matrix(dummy_X, encoded_y):

    X_train, X_test, Y_train, Y_test = train_test_split(dummy_X, encoded_y, test_size=0.25)
    D_train = xgb.DMatrix(X_train, label=Y_train)
    D_test = xgb.DMatrix(X_test, label=Y_test)
    xgb_matrix = {}
    xgb_matrix['train'] = D_train
    xgb_matrix['test'] = D_test

    return xgb_matrix

@pytest.fixture(scope="session", name="xgb_model")
def fixture_xgb_model(xgb_matrix, encoded_y):

    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 1
    param['objective'] = 'multi:softprob'
    param['num_class'] = len(np.unique(encoded_y))
    num_round = 10

    xgb_model = xgb.train(param, xgb_matrix['train'], num_round)

    return xgb_model

@pytest.fixture(scope="session", name="save_xgb_model_path")
def fixture_save_xgb_model(xgb_model):
    filename = 'resources/models/gbr/gbr_model.pkl'
    pickle.dump(gbr_model, open(filename, 'wb'))
    return filename

@pytest.fixture(scope="session", name="pls_pred")
def fixture_pls_pred(X, y, pls_model):
    # Predict using the model
    pls_pred = pls_model.predict(X)
    return pls_pred

@pytest.fixture(scope="session", name="gbr_pred")
def fixture_gbr_pred(X, y, gbr_model):
    # Predict using the model
    gbr_pred = gbr_model.predict(X)
    return gbr_pred

# Create a volume to load models from the host machine to the Docker container
@pytest.fixture(scope='session', autouse=True)
def volume():
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

        # Create the model volume
        model_dir = os.path.abspath('resources/models')
        print(model_dir)
        subprocess.check_call(
            'docker volume create --name model_volume --opt type=none '
            '--opt device={} --opt o=bind'.format(model_dir).split())
        yield model_dir
    finally:
        subprocess.check_call('docker volume rm model_volume'.split())

@pytest.fixture(scope='module', autouse=True)
def container():
    try:
        command = (
            "docker run --name coa-inference-test -p 8080:8080 " +
            "-e SAGEMAKER_MULTI_MODEL=true " +
            " --mount type=volume,source=model_volume,target=/opt/ml/model " + IMAGE_NAME +
            " serve"
        )

        print(command)
        proc = subprocess.Popen(command.split(), stdout=sys.stdout, stderr=subprocess.STDOUT)

        attempts = 0
        while attempts < 5:
            time.sleep(3)
            try:
                requests.get(PING_URL)
                break
            except:  # noqa: E722
                attempts += 1
                pass
        yield proc.pid
    finally:
        subprocess.check_call("docker rm -f coa-inference-test".split())


def make_list_model_request():
    response = requests.get(MODELS_URL)
    return response.status_code, json.loads(response.content.decode("utf-8"))


def make_load_model_request(data, content_type="application/json"):
    headers = {"Content-Type": content_type}
    print(data)
    response = requests.post(MODELS_URL, data=data, headers=headers)
    return response.status_code, json.loads(response.content.decode("utf-8"))


def make_unload_model_request(model_name):
    response = requests.delete(DELETE_MODEL_URL.format(model_name))
    return response.status_code, json.loads(response.content.decode("utf-8"))


def make_invocation_request(model_name, data, content_type="application/json"):
    headers = {"Content-Type": content_type}
    response = requests.post(INVOCATION_URL.format(model_name), data=data, headers=headers)
    return response.status_code, json.loads(response.content.decode("utf-8"))


def test_ping():
    res = requests.get(PING_URL)
    print(res)
    assert res.status_code == 200

def test_list_models_default():
    code, models = make_list_model_request()
    assert code == 200
    assert models["models"] == []

def test_pls_model_pickle(save_pls_model_path, X, y, pls_pred):
    # load the model from disk
    loaded_model = pickle.load(open(save_pls_model_path, 'rb'))
    preds = loaded_model.predict(X)

    assert np.equal(preds, pls_pred).all()

def test_gbr_model_pickle(save_gbr_model_path, X, y, gbr_pred):
    # load the model from disk
    loaded_model = pickle.load(open(save_gbr_model_path, 'rb'))
    preds = loaded_model.predict(X)

    assert np.equal(preds, gbr_pred).all()

def test_load_models():
    data1 = {"model_name": "PLS", "url": "/opt/ml/model/pls"}
    code1, content1 = make_load_model_request(data=json.dumps(data1))
    assert code1 == 200
    assert content1["status"] == "Workers scaled"

    code2, content2 = make_list_model_request()
    assert code2 == 200
    assert content2["models"] == [{"modelName": "PLS", "modelUrl": "/opt/ml/model/pls"}]

    data2 = {"model_name": "GBR", "url": "/opt/ml/model/gbr"}
    code3, content3 = make_load_model_request(data=json.dumps(data2))
    assert code3 == 200
    assert content3["status"] == "Workers scaled"

    code4, content4 = make_list_model_request()
    assert code4 == 200
    assert content4["models"] == [
        {"modelName": "GBR", "modelUrl": "/opt/ml/model/gbr"},
        {"modelName": "PLS", "modelUrl": "/opt/ml/model/pls"},
    ]
    print(content4)

def test_unload_models():
    code1, content1 = make_unload_model_request("PLS")
    assert code1 == 200
    assert content1["status"] == 'Model "PLS" unregistered'

    code2, content2 = make_list_model_request()
    assert code2 == 200
    assert content2["models"] == [{"modelName": "GBR", "modelUrl": "/opt/ml/model/gbr"}]


def test_load_non_existing_model():
    data1 = {"model_name": "banana", "url": "/banana"}
    code1, content1 = make_load_model_request(data=json.dumps(data1))
    assert code1 == 404


def test_unload_non_existing_model():
    # dummy_model_1 is already unloaded
    code1, content1 = make_unload_model_request("PLS")
    assert code1 == 404


def test_load_model_multiple_times():
    # resnet_18 is already loaded
    data = {"model_name": "GBR", "url": "/opt/ml/model/gbr"}
    code3, content3 = make_load_model_request(data=json.dumps(data))
    assert code3 == 409

def test_invocation(gbr_pred, gbr_model):
    print(os.getcwd())
    csv_file = "resources/genopheno/data/training/genopheno.csv"
    df_csv = pd.read_csv(csv_file)
    df_csv = df_csv.iloc[0:10,1:]
    df_json = df_csv.to_json()

    code, predictions = make_invocation_request("GBR", df_json)
    print(gbr_pred[0:10])
    print(gbr_model.predict(df_csv.iloc[0:10,:]))
    assert code == 200
    assert predictions == gbr_pred[0:2].tolist()