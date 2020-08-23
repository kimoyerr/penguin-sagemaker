"""
This module contains functions to run the XGBoost model on the Penguin data

TODO:
"""

# Built-in libraries
import sys
import os
from os.path import abspath, dirname
import logging


# External libraries
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score
import xgboost as xgb

# Set Loggers
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Functions to pre-process data
def dummify_X(X, cat_columns):
    """Function to load the feature set and convert categorical columns into one-hot-encoded columns.
    The original categorical columns will be removed and the new columns will be appended at the end

     Parameters
    ----------
    X : pandas.DataFrame
        The dataframe for just the features
    cat_columns : list of str
        The columns that have to be converted from categorical to one-hot-encodings

    Returns
    -------
    pandas.DataFrame
        The dataframe with the original categorical columns removed and the new one-hot-encoded columns at the end

    """

    dummy_X = X.copy()
    for col in cat_columns:
        dummy_X = pd.concat([dummy_X.drop(col, axis=1), pd.get_dummies(dummy_X[col])], axis=1)

    return dummy_X

def encode_y(y):
    """Function to load the categorical labels and convert them to integer labels

     Parameters
    ----------
    y : pandas.Series
        The values of the categorical labels

    Returns
    -------
    sklearn.preprocessing.LabelEncoder
        The encoder to transform the categorical variables to integer labels fit on the data set

    """

    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)
    keys = encoder.classes_
    values = encoder.transform(encoder.classes_)
    dictionary = dict(zip(keys, values))

    return encoder

def create_xgb_matrix(dummy_X, y, encoder, test_fraction = 0.25):
    """Function to create a dictionary of Dmatices for XGBoost training and evaluation

     Parameters
    ----------
    dummy_X : pandas.DataFrame
        The dataframe with categorical variables replaced by one-hot-encoded labels

    y : pandas.Series
        The values of the label to be predicted

    test_fraction : float
        The fraction of the dataset to be used for testing or evaluation after training

    encoder : sklearn.preprocessing.LabelEncoder
        The encoder that has been trained to transform the categorical labels to integer labels

    Returns
    -------
    dict of {str : xgboost.DMatrix}
        The dictionary of xgboost Dmatrices

    """
    if test_fraction!=0:
        encoded_y = encoder.transform(y)
        logger.info(encoded_y)
        X_train, X_test, Y_train, Y_test = train_test_split(dummy_X, encoded_y, test_size=test_fraction)
        D_train = xgb.DMatrix(X_train, label=Y_train)
        D_test = xgb.DMatrix(X_test, label=Y_test)
        xgb_matrix = {}
        xgb_matrix['train'] = D_train
        xgb_matrix['test'] = D_test
    else:
        encoded_y = encoder.transform(y)
        logger.info(encoded_y)
        D_train = xgb.DMatrix(dummy_X, label=encoded_y)
        xgb_matrix = {}
        xgb_matrix['train'] = D_train

    return xgb_matrix

def fit_xgb(xgb_matrix, num_rounds, params):
    """Function to fit the XGboost model

     Parameters
    ----------
    xgb_matrix : dict of {str : xgboost.DMatrix}
        The dictionary of xgboost Dmatrices with one matrix for training and the other for evaluation

    num_rounds : int
        The number of boosting rounds

    params : dict-like
        The parameters to be used for training the models

    Returns
    -------
    xgboost.Booster
        The xgboost trained booster model

    """

    xgb_model = xgb.train(xgb_matrix, num_rounds, params)

    return xgb_model

def predict_xgb(xgb_model, xgb_matrix):
    """Function to fit the XGboost model

     Parameters
    ----------
    xgb_model : object of {str : xgboost.Booster}
        The xgboost Booster object
    xgb_matrix : object of {str : xgboost.DMatrix}
        The xgboost dmatrix object of data

    Returns
    -------
    numpy.array
        The array of predictions

    """

    xgb_preds = xgb_model.predict(xgb_matrix)
    xgb_preds = np.asarray([np.argmax(p) for p in xgb_preds])

    return xgb_preds

def xgb_cv(xgb_matrix, num_rounds, num_folds, params):
    """Function to perform cross-validation

     Parameters
    ----------
    xgb_matrix : dict of {str : xgboost.DMatrix}
        The dictionary of xgboost Dmatrices with only one matrix for training and no test matrix

    num_rounds : int
        The number of boosting rounds

    num_folds : int
        The number of data folds to use for cross-validation

    params : dict-like
        The parameters to be used for training the models

    Returns
    -------
    Pandas DataFrame
        The dataframe of results from cross-validation

    """

    cv_res = xgb.cv(params, xgb_matrix['train'], num_boost_round=num_rounds, nfold=num_folds)

    return cv_res
