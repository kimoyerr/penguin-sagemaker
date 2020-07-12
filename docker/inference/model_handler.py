"""
ModelHandler defines an example model handler for load and inference requests for GBR and other coa models
Can handle multiple models on the same server. Each model will be identified by the context object that is passed along
with the data from the multi-model-server

TODO: Check the column names in the input dataframe for prediction and reorder columns as needed
TODO: Implement custom pickling of models that also stores the column order for the training data
"""
# From: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/multi_model_bring_your_own/container/model_handler.py


from collections import namedtuple
import glob
import json
import logging
import os
import re
import time
import pickle
from io import StringIO

import mxnet as mx
import numpy as np
import pandas as pd

from sagemaker_inference import content_types, decoder, encoder, utils
# Import local custom models
from learn.modeling import GBRModel

class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.mx_model = None
        self.shapes = None

    def get_model_files_prefix(self, model_dir):
        """Takes the model directory parameter from the context object and returns the string for the path of model files.
        Args:
            model_dir (str): Path to the directory with model artifacts
        Returns:
            (obj): Prefix string for model artifact files
        """

        # Get the pickled file.
        pkl_file_suffix = ".pkl"
        checkpoint_prefix_regex = "{}/*{}".format(model_dir,
                                                  pkl_file_suffix)  # Ex output: /opt/ml/models/dummy_1/*.pkl
        checkpoint_prefix_filename = glob.glob(checkpoint_prefix_regex)[
            0]  # Ex output: /opt/ml/models/dummy_1/dummy_model.pkl
        checkpoint_prefix = os.path.basename(checkpoint_prefix_filename).split(pkl_file_suffix)[
            0]  # Ex output: dummy_model
        logging.info("Prefix for the model artifacts: {}".format(checkpoint_prefix))
        return checkpoint_prefix


    def initialize(self, context):
        """Initializes a model depending on the context information provided. 
            This will be done only once per model.
        Args:
            context (obj): Initial context that contains model server system properties
        Returns:
            (None): 
        """

        # Parse the context object
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")
        logging.info(model_dir)
        gpu_id = properties.get("gpu_id")

        # Test pickled model
        checkpoint_prefix = self.get_model_files_prefix(model_dir)
        model_file_path = os.path.join(model_dir, "{}{}".format(checkpoint_prefix, ".pkl"))
        self.mx_model = pickle.load(open(model_file_path, 'rb'))
        logging.info('model loaded')
        logging.info(self.mx_model)

    def preprocess(self, request, context):
        """Takes request data and de-serializes the data into an object for prediction.
            When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
            the model server receives two pieces of information:
                - The request Content-Type, for example "application/json"
                - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
            The input_fn is responsible to take the request data and pre-process it before prediction.
        Args:
            input_data (obj): The request data.
            content_type (str): The context for the request
        Returns:
            (obj): The data-frame ready for prediction.
        """
        logging.info('Inside the Pre-processing Function')
        request_processor = context.request_processor[0]
        request_property = request_processor.get_request_properties()
        content_type = utils.retrieve_content_type_header(request_property)
        logging.info(content_type)

        bytes_data = request[0].get('body')
        s = str(bytes_data, 'utf-8')
        logging.info(s)
        data = StringIO(s)
        df = pd.io.json.read_json(data, orient='columns')
        logging.info(df)

        return df

    def inference(self, model_input):
        """Takes the pre-processed data-frame and perfroms predictions on it using the current model
        Args:
            imodel_input (Pandas data-frame): The pre-processed input data as a data-frame
        Returns:
            (numpy array): The predictions from the model
        """

        logging.info('Inside the Inference Function')
        preds = self.mx_model.predict(model_input)
        logging.info(preds)
        return preds


    def postprocess(self, inference_output):
        """Takes the predictions from the model and does any post-processing.
            The final data needs to be a list otherwise will throw an error.
        Args:
            inference_output (numpy array): Predictions from the model
        Returns:
            (list): List of post-processed predictions
        """

        logging.info('Inside the Post-processing Function')
        logging.info(inference_output)
        json_out = json.dumps(inference_output.tolist())
        return json_out

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        """Takes the data and context from the server and puts them through a series of steps and returns the data
            Can add any other functions to the ones listed here
        Args:
            data (list of bytes array objects): The data passed to the server
        Returns:
            context (object): Other metadata associated with the data including what model should be used for predictions
        """

        model_input = self.preprocess(data, context)
        model_out = self.inference(model_input)
        return_data = self.postprocess(model_out)

        return [return_data] # Has to be a list to make sure it is JSON serializable. Otherwise will throw an error

# Start the ModelHandler
_service = ModelHandler()

# The handle function passed to the multi-model-server
def handle(data, context):
    if not _service.initialized:
        logging.info(context)
        logging.info(data)

        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)