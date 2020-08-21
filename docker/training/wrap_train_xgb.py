#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to wrap the model.fit into a script similar to Amazon sagemaker training scripts
"""

# Generic/Built-in Imports
import ast
import argparse
import os
import time
from pathlib import Path

# Other Libs
import numpy as np
import pandas as pd
import joblib as joblib

# Local Paths and Modules
# sys.path.append(os.path.abspath('.'))
from penguin.penguin_xgb import dummify_X, encode_y, create_xgb_matrix, fit_xgb

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
    logger.info(args)

    # Load data
    logger.info("Loading Data")

    df = pd.read_csv(Path(args.data_dir, args.train_file), index_col=0)
    logger.info("Input Data Shape:")
    logger.info(df.shape)

    # Pre-process data
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    logger.info("shape of features={}, shape of targets={}".format(X.shape, y.shape))
    # Convert categorical columns to one-hot encodings
    cat_columns = args.categorical_columns.split(',')
    dummy_X = dummify_X(X, cat_columns)
    # Convert the categorical labels in the target to integer labels
    label_encoder = encode_y(y)
    encoded_y = label_encoder.transform(y)

    # Create the xgboost matrices for training and evaluation
    xgb_matrix = create_xgb_matrix(dummy_X, y, label_encoder, args.test_fraction)
    logger.info('Data sets present:')
    logger.info(xgb_matrix.keys())

    # Create the xgboost model and train
    logger.info("Training model")
    train_params = {"max_depth": args.max_depth,
                    "eta": args.eta,
                    "nthread": args.nthread,
                    "objective": args.objective,
                    "num_class": len(np.unique(encoded_y))}
    model = fit_xgb(train_params, xgb_matrix['train'], args.num_rounds)
    logger.info('Finished Training')

    # Save the model
    logger.info(args.model_dir)
    model.save_model(os.path.join(args.model_dir, "penguin_xgb_model.json"))
    logger.info(os.listdir(args.model_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Data parameters
    parser.add_argument('--train-file', type=str, default=None, metavar='TF',
                        help='The file in S3 to use for training (default: None)')
    parser.add_argument('--categorical-columns', type=str, default=None, metavar='CC',
                        help='The columns in the dataframe to convert to one-hot-encodings (default: None)')
    parser.add_argument('--test-fraction', type=float, default=0.25, metavar='TF',
                        help='The fraction of the total data to be used for model evaluation or testing (default: 0.25)')

    # Model parameters
    parser.add_argument('--max-depth', type=int, default=2, metavar='MD',
                        help='The maximum number of splits allowed in each tree (default: 2)')
    parser.add_argument('--eta', type=float, default=1, metavar='ETA',
                        help='The learning rate or shrinkage parameter to use to update the feature weights after each boosting round (default: 0.9])')
    parser.add_argument('--objective', type=str, default='multi:softprob', metavar='OB',
                        help='The objective function to maximize (default: multi:softprob)')

    # Train parameters
    parser.add_argument('--nthread', type=int, default=1, metavar='NT',
                        help='number of processors to use (default: 1)')
    parser.add_argument('--num-rounds', type=int, default=5, metavar='NR',
                        help='number of boosting rounds to use (default: 5)')


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