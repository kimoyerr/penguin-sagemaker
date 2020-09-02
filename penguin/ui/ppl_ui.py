"""
This script creates a page for probabilistic programming using Stan
"""

# Internal libraries
from os.path import abspath, dirname
import io
import uuid

# External libraries
import streamlit as st
import numpy as np
import pandas as pd
import scipy
import mlflow
import pystan
import arviz
from scipy.stats import mode
from sklearn.model_selection import train_test_split

# Sagemaker and AWS libraries
import boto3
from sagemaker import LocalSession

# Internal libraries
from penguin.penguin_xgb import dummify_X
from penguin.penguin_xgb import encode_y
from penguin.class_utils.conf_mat_helpers import plot_confusion_matrix

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Some constants
IMAGE_NAME = 'penguin-xgb-training'
REGION_NAME = 'us-west-2'
ROLE = 'arn:aws:iam::784420883498:role/service-role/AmazonSageMaker-ExecutionRole-20200313T094543'
BUCKET_NAME = 'ky-blogs'
MODEL_SAVE_OBJ = 'penguins/'
MODEL_SAVE_PATH = 's3://' + BUCKET_NAME + '/' + MODEL_SAVE_OBJ

# Create an S3 client
s3 = boto3.resource('s3')

# Change working directory to 'test' directory
project_dir = dirname(dirname(dirname(abspath(__file__))))
print(project_dir)

@st.cache
def create_mlflow_exp():
    # Create a new experiment with unique ID
    exp_uniq_name = 'exp_' + uuid.uuid4().hex[:6].upper()
    mlflow.set_experiment(exp_uniq_name)

    return exp_uniq_name


def page(state):

    # Streamlit session state
    print(state.mlflow_res.shape)

    # Delete any existing mlflow experiments naned '0'
    try:
        mlflow.delete_experiment('0')
    except:
        pass

    # Create a new experiment with unique ID
    exp_uniq_name = create_mlflow_exp()

    # Title and Description
    st.title('Probabilistic Programming')
    st.markdown("""
        This page guides you through the selection of algorithms and hyperparameters for Multi Logit models using Bayesian Inference

        **Start by loading the data**.
        """)

    data_file = st.file_uploader("Upload data...", type="csv", key='train_data')
    try:
        text_io = io.TextIOWrapper(data_file)
    except:
        pass
    if data_file is not None:
        W = pd.read_csv(data_file, index_col=False)
        y = W.iloc[:, 0]
        X = W.iloc[:, 1:]
        st.write(W)

    # Options to select Model
    st.subheader('Select Model')
    model_types = ['Multi-Logit']
    sel_model = st.radio('Which model to use for training?', model_types)

    params = {}
    if sel_model=='Multi-Logit':
        # Get parameters for Multi-Logit
        params['num-chains'] = st.sidebar.number_input(label="Number of chains for sampling", min_value=1, max_value=4, value=4,
                                              step=1)
        params['num-iters'] = st.sidebar.number_input(label="Number of iterations for sampling", min_value=100,
                                                      max_value=1000, value=1000, step=100)
        params['num-warmup-iters'] = st.sidebar.number_input(label="Number of iterations for warmup", min_value=100,
                                                      max_value=1000, value=500, step=100)
        params['max-tree-depth'] = st.sidebar.number_input(label="Maximum tree depth for the NUTS sampler", min_value=10,
                                                             max_value=20, value=10, step=1)
        sel_sampler = st.radio('Which sample to use?', ['NUTS', 'HMC'])


    # Sagemaker Training options
    instance_types = ['local', 'ml.m5.large', 'ml.m5.xlarge', 'ml.m5.4xlarge', 'ml.m5.24xlarge', 'ml.g4dn.xlarge', 'ml.g4dn.4xlarge', 'ml.g4dn.16xlarge']
    sagemaker_instance = st.sidebar.selectbox('Instance type for Sagemaker training', instance_types)

    if len(params) > 0:
        samp_submit = st.sidebar.button('Run Sampling')
        if samp_submit:
            if sel_model == 'Multi-Logit':
                model_title = 'Multi-Logit'
                image_name = IMAGE_NAME
                model_name_suffix = 'penguin_xgb_model.json'

            if sagemaker_instance == 'local':

                # Data prep
                W = W.iloc[:, 1:]
                dummy_X = dummify_X(W.iloc[:,1:], cat_columns=['island', 'sex'])
                y = W.iloc[:, 0]
                encoder, encoder_dict = encode_y(y)

                # Remove nan rows
                nan_rows = np.where(dummy_X.isnull().any(axis=1))
                X = dummy_X.drop(dummy_X.index[nan_rows], inplace=False)
                y = y.drop(y.index[nan_rows], inplace=False)

                # Split train and test data with the same random state
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

                data = {
                    'N': X_train.shape[0],
                    'N2': X_test.shape[0],
                    'D': X_train.shape[1],
                    'K': len(np.unique(y_train)),
                    'y': encoder.transform(y_train) + 1,
                    'x': X_train,
                    'x_new': X_test,
                }

                # Model specifications
                model = pystan.StanModel(file='penguin/prob_prog/multinomial.stan')
                if sel_sampler =='NUTS':
                    fit = model.sampling(data=data, iter=params['num-iters'], chains=params['num-chains'],
                                         algorithm=sel_sampler, control=dict(max_treedepth=params['max-tree-depth']))
                fit_samp = fit.extract(permuted=True)
                np.save('data/pystan_results/beta_posterior_NUTS_max_tree_depth_15.npy', fit_samp['beta'])

                # Plots
                tmp = fit.stansummary().replace('\n', '\n\t') # For streamlit to render the table better
                st.write(tmp)
                arviz.plot_trace(fit)
                st.pyplot()

                # Model predictions for Training
                X_np_train = X_train.to_numpy()
                X_np_test = X_test.to_numpy()
                preds_train = np.empty([X_np_train.shape[0], fit_samp['beta'].shape[0]])
                preds_test = np.empty([X_np_test.shape[0], fit_samp['beta'].shape[0]])
                for i in range(fit_samp['beta'].shape[0]):
                    # Train
                    scipy.special.softmax(X_np_train.dot(fit_samp['beta'][i, :]), axis=1)
                    preds_train[:, i] = np.argmax(scipy.special.softmax(X_np_train.dot(fit_samp['beta'][0, :]), axis=1),
                                                  axis=1)
                    # Test
                    scipy.special.softmax(X_np_test.dot(fit_samp['beta'][i, :]), axis=1)
                    preds_test[:, i] = np.argmax(scipy.special.softmax(X_np_test.dot(fit_samp['beta'][0, :]), axis=1),
                                                  axis=1)

                # Get consensus predictions from all samples
                cons_preds_train = mode(preds_train, axis=1)[0]
                cons_preds_test = mode(preds_test, axis=1)[0]

                np.savetxt('data/pystan_results/preds_posterior_NUTS_max_tree_depth_10.csv', preds, delimiter=',')
                plot_confusion_matrix(encoder.transform(y_train), cons_preds_train[:,0],
                classes = np.asarray(list(encoder_dict.keys())),
                title = 'Confusion matrix, without normalization')
                plot_confusion_matrix(encoder.transform(y_test), cons_preds_test[:,0],
                classes = np.asarray(list(encoder_dict.keys())),
                title = 'Confusion matrix, without normalization')
