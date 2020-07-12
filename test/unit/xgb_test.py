#!/usr/bin/python
"""
This script performs unit tests for the XGBoost model

"""

# Built-in libraries
import sys
import os
from os.path import abspath, dirname

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

# Change working directory to 'test' directory
cwd_dir = dirname(dirname(abspath(__file__)))
os.chdir(cwd_dir)
print(os.getcwd())


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


def test_dummify(dummy_X, encoded_y):
    print(encoded_y)
    assert dummy_X.shape[1] == 9

def test_data_split(xgb_matrix, dummy_X, encoded_y):
    print(xgb_matrix['train'])
    assert len(xgb_matrix['train'].get_label()) == len(encoded_y)*0.75
    assert xgb_matrix['train'].num_col() == dummy_X.shape[1]
    assert xgb_matrix['test'].num_col() == dummy_X.shape[1]

def test_xgb_model_predictions(xgb_model, xgb_matrix):
    train_preds = xgb_model.predict(xgb_matrix['train'])
    train_preds = np.asarray([np.argmax(p) for p in train_preds])
    print(precision_score(xgb_matrix['train'].get_label(), train_preds, average='macro'))
    print(accuracy_score(xgb_matrix['train'].get_label(), train_preds))

    test_preds = xgb_model.predict(xgb_matrix['test'])
    test_preds = np.asarray([np.argmax(p) for p in test_preds])
    print(precision_score(xgb_matrix['test'].get_label(), test_preds, average='macro'))
    print(accuracy_score(xgb_matrix['test'].get_label(), test_preds))

    assert precision_score(xgb_matrix['train'].get_label(), train_preds, average='macro') > 0.9
    assert accuracy_score(xgb_matrix['train'].get_label(), train_preds) > 0.9



