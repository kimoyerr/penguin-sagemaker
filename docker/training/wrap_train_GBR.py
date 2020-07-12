#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to wrap the model.fit into a script similar to Amazon sagemaker training scripts
"""

# Generic/Built-in Imports
import ast
import argparse
import os
from pathlib import Path

# Other Libs
import numpy as np
import pandas as pd
import joblib as joblib

# Local Paths and Modules
# sys.path.append(os.path.abspath('.'))
from algorithms.learn.modeling import GBRModel

print(GBRModel)
# Set Loggers
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Function to convert string arguments to boolean if needed
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Training function
def _train(args):
    print(args)

    # Load data
    logger.info("Loading Data")
    W = pd.read_csv(Path(args.data_dir, args.train_file))
    print(W.shape)
    y = W.iloc[:, 0]
    X = W.iloc[:, 1:]
    print(X.shape, y.shape)

    #Create model
    model = GBRModel(args.quadraticX, n_estimators=args.n_estimators, max_depth=args.max_depth,
                     learning_rate=args.learning_rate)
    print('train')

    # Train the model
    logger.info("Training model")
    model.fit(X, y)
    print('Finished Training')
    print('yes')

    # Save the model
    print(args.model_dir)
    joblib.dump(model, os.path.join(args.model_dir, "GBRModel.joblib"))


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "PLSR.joblib"))
    return clf


def predict_fn(input_data, model):

    # logger.info('Generating text based on input parameters.')
    yp = model.predict(input_data)

    return yp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Data parameters
    parser.add_argument('--train-file', type=str, default=None, metavar='TF',
                        help='The file in S3 to use for training (default: None)')

    # Model parameters
    parser.add_argument('--quadraticX', type=str2bool, nargs='?', const=True, default=False,
                        help='Should the quadratic terms for genotype be included (default: False')
    parser.add_argument('--n-estimators', type=int, default=100, metavar='NES',
                        help='The number of trees to use for the estimation (default: 100)')
    parser.add_argument('--max-depth', type=int, default=3, metavar='MD',
                        help='The maximum number of branches allowed per tree (default: 3)')

    # Train parameters
    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=0.1, metavar='LR',
                        help='The learning rate to use (default: 0.1)')


    # OUtput and Debug parameters
    parser.add_argument('--dist-backend', type=str, default='gloo',
                        help='distributed backend (default: gloo)')
    parser.add_argument('--model-name', type=str, default='model', metavar='MO',
                        help='name for the model')
    parser.add_argument('--save-all-models', type=int, default=0, metavar='SAM',
                        help='Indicates whether a checkpoint is saved after each epoch.If 0, only the best performing model is saved (default: 0)')
    parser.add_argument('--debug-steps', type=int, default=100, metavar='DS',
                        help='number of steps to save the debug output')
    parser.add_argument('--debug-on', type=int, default=1, metavar='D)',
                        help='should debug and hooks be turned on')
    parser.add_argument('--debug-dir', type=str, default=None, metavar='DD',
                        help='s3 directory to store the debug outputs')


    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--input-dir', type=str, default=os.environ['SM_INPUT_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])


    #Train
    _train(parser.parse_args())