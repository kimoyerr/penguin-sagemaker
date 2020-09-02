#!/usr/bin/python
"""
This script performs unit tests for the PyStan models

"""

# Built-in libraries
import sys
import os
from os.path import abspath, dirname

# External libraries
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import pytest
import pystan

# Logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Change working directory to 'test' directory
cwd_dir = dirname(dirname(abspath(__file__)))
os.chdir(cwd_dir)
print(os.getcwd())


def test_multinomial(dummy_X, y, encoder):
    nan_rows = np.where(dummy_X.isnull().any(axis=1))
    X = dummy_X.drop(dummy_X.index[nan_rows], inplace=False)
    Y = y.drop(y.index[nan_rows], inplace=False)
    data = {
        'N': X.shape[0],
        'N2': X.shape[0],
        'D': X.shape[1],
        'K': len(np.unique(Y)),
        'y': encoder.transform(Y)+1,
        'x': X,
        'x_new': X,
    }
    model = pystan.StanModel(file='unit/multinomial.stan')
    fit = model.sampling(data=data, iter=1000, chains=1)

    assert fit['beta'].shape[0] == 500
    assert fit['beta'].shape[1] == 9
    assert fit['beta'].shape[2] == 3


