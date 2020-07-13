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


@pytest.fixture(scope="session", name="xgb_model")
def fixture_xgb_model(xgb_matrix, y, encoder):

    encoded_y = encoder.transform(y)
    param = {'max_depth': 2, 'eta': 1}
    param['nthread'] = 1
    param['objective'] = 'multi:softprob'
    param['num_class'] = len(np.unique(encoded_y))
    num_round = 10

    xgb_model = xgb.train(param, xgb_matrix['train'], num_round)

    return xgb_model


def test_dummify(dummy_X, y, encoder):

    encoded_y = encoder.transform(y)
    print(encoded_y)
    assert dummy_X.shape[1] == 9

def test_data_split(xgb_matrix, dummy_X, y, encoder):

    encoded_y = encoder.transform(y)
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


def test_save_and_load_xgb_model(xgb_model, xgb_matrix):
    xgb_model.save_model('resources/models/xgb_model.json')
    loaded_model = xgb.Booster()
    loaded_model.load_model('resources/models/xgb_model.json')

    assert np.equal(xgb_model.predict(xgb_matrix['test']), loaded_model.predict(xgb_matrix['test'])).all()
