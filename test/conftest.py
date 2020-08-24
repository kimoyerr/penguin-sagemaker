# Built-in libraries
import sys
import os
from os.path import abspath, dirname

# External libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pytest
import xgboost as xgb

# Logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Fixtures
@pytest.fixture(scope="session", name="df")
def fixture_df():
    df = pd.read_csv('resources/data/penguins.csv', index_col=0)
    return df


@pytest.fixture(scope="session", name="X")
def fixture_X(df):
    X = df.iloc[:, 1:]
    return X


@pytest.fixture(scope="session", name="y")
def fixture_y(df):
    y = df.iloc[:, 0]
    return y


@pytest.fixture(scope="session", name="dummy_X")
def fixture_dummify_X(X):
    sel_columns = ['island', 'sex']
    dummy_X = X.copy()
    for col in sel_columns:
        dummy_X = pd.concat([dummy_X.drop(col, axis=1), pd.get_dummies(dummy_X[col])], axis=1)

    return dummy_X


@pytest.fixture(scope="session", name="encoder")
def fixture_encode_y(y):
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)
    keys = encoder.classes_
    values = encoder.transform(encoder.classes_)
    dictionary = dict(zip(keys, values))
    logger.debug(dictionary)

    return encoder


@pytest.fixture(scope="session", name="xgb_matrix")
def fixture_xgb_matrix(dummy_X, y, encoder):
    encoded_y = encoder.transform(y)
    X_train, X_test, Y_train, Y_test = train_test_split(dummy_X, encoded_y, test_size=0.25)
    D_train = xgb.DMatrix(X_train, label=Y_train)
    D_test = xgb.DMatrix(X_test, label=Y_test)
    xgb_matrix = {}
    xgb_matrix['train'] = D_train
    xgb_matrix['test'] = D_test

    return xgb_matrix


@pytest.fixture(scope="session", name="xgb_matrix_cv")
def fixture_xgb_matrix_cv(dummy_X, y, encoder):
    encoded_y = encoder.transform(y)
    D_train = xgb.DMatrix(dummy_X, label=encoded_y)
    xgb_matrix = {}
    xgb_matrix['train'] = D_train

    return xgb_matrix
