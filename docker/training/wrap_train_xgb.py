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
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score

# Local Paths and Modules
# sys.path.append(os.path.abspath('.'))
from penguin.penguin_xgb import dummify_X, encode_y, create_xgb_matrix, fit_xgb, predict_xgb, xgb_cv

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
    # Change args test fraction if cv is turned on
    if args.do_cv == 1:
        args.test_fraction = 0

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
    train_params = {"max_depth": args.max_depth,
                    "eta": args.eta,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "nthread": args.nthread,
                    "objective": args.objective,
                    "num_class": len(np.unique(encoded_y))}
    if args.do_cv!=1:
        logger.info("Training model")
        model = fit_xgb(train_params, xgb_matrix['train'], args.num_rounds)
        logger.info('Finished Training')

        # Save the model
        logger.info(args.model_dir)
        model.save_model(os.path.join(args.model_dir, "penguin_xgb_model.json"))
        logger.info(os.listdir(args.model_dir))
        # Save the results
        train_preds = predict_xgb(model, xgb_matrix['train'])
        test_preds = predict_xgb(model, xgb_matrix['test'])
        train_precision_score = precision_score(xgb_matrix['train'].get_label(), train_preds, average='macro')
        train_accuracy_score = accuracy_score(xgb_matrix['train'].get_label(), train_preds)
        test_precision_score = precision_score(xgb_matrix['test'].get_label(), test_preds, average='macro')
        test_accuracy_score = accuracy_score(xgb_matrix['test'].get_label(), test_preds)
        xgb_metrics = np.asarray([train_precision_score, train_accuracy_score, test_precision_score, test_accuracy_score])
        logger.info(args.output_dir)
        xgb_matrix['train'].save_binary(os.path.join(args.output_dir, "dmatrix_train.data"))
        xgb_matrix['test'].save_binary(os.path.join(args.output_dir, "dmatrix_test.data"))
        np.savetxt(os.path.join(args.output_dir, "train_preds.csv"), train_preds, delimiter=",")
        np.savetxt(os.path.join(args.output_dir, "test_preds.csv"), test_preds, delimiter=",")
        np.savetxt(os.path.join(args.output_dir, "all_metrics.csv"), xgb_metrics, delimiter=",")
        logger.info(os.listdir(args.output_dir))
    else:
        logger.info("Cross-Validating")
        cv_res = xgb_cv(xgb_matrix, args.num_rounds, args.num_folds, train_params)
        logger.info('Finished Cross-Validation')
        # Save the results
        logger.info(args.output_dir)
        cv_res.to_csv(os.path.join(args.output_dir, "penguin_xgb_cv.csv"), index=False)
        logger.info(os.listdir(args.output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Data parameters
    parser.add_argument('--train-file', type=str, default=None, metavar='TF',
                        help='The file in S3 to use for training (default: None)')
    parser.add_argument('--categorical-columns', type=str, default=None, metavar='CC',
                        help='The columns in the dataframe to convert to one-hot-encodings (default: None)')
    parser.add_argument('--test-fraction', type=float, default=0.25, metavar='TF',
                        help='The fraction of the total data to be used for model evaluation or testing (default: 0.25)')
    parser.add_argument('--do-cv', type=int, default=0, metavar='CV',
                        help='Should cross-validation be performed instead of training and testing? (default: 0)')


    # Model parameters
    parser.add_argument('--num-rounds', type=int, default=10, metavar='NR',
                        help='The number of rounds or number of trees to use for boosting (default: 10)')
    parser.add_argument('--max-depth', type=int, default=2, metavar='MD',
                        help='The maximum number of splits allowed in each tree (default: 2)')
    parser.add_argument('--eta', type=float, default=1, metavar='ETA',
                        help='The learning rate or shrinkage parameter to use to update the feature weights after each boosting round (default: 0.9])')
    parser.add_argument('--subsample', type=float, default=1.0, metavar='SS',
                        help='Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting (default: 0.75])')
    parser.add_argument('--colsample-bytree', type=float, default=1.0, metavar='CST',
                        help='Subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed (default: 0.75])')
    parser.add_argument('--num-folds', type=int, default=10, metavar='NF',
                        help='The number of folds to use in cross-validation (default: 10)')
    parser.add_argument('--objective', type=str, default='multi:softprob', metavar='OB',
                        help='The objective function to maximize (default: multi:softprob)')

    # Train parameters
    parser.add_argument('--nthread', type=int, default=1, metavar='NT',
                        help='number of processors to use (default: 1)')

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