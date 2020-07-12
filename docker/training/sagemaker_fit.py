#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for work with sagemaker containers
"""

# Imports
import boto3
import sagemaker
from sagemaker.estimator import Estimator


def crete_estimator(image_name, region_name, instance_count, instance_type, model_save_path, **hyperparams):
    """Function to create the estimator for sagemaker training.

     Parameters
    ----------
    image_name : str
        The name of the image to use for training

    Returns
    -------
    Sagemaker Estimator object created from the container

    """

    estimator = Estimator(
        role='arn:aws:iam::784420883498:role/service-role/AmazonSageMaker-ExecutionRole-20200313T094543',
        sagemaker_session=sagemaker.Session(boto3.session.Session(region_name=region_name)),
        train_instance_count=instance_count,
        train_instance_type=instance_type,
        image_name=image_name,
        output_path=model_save_path,
        hyperparameters=hyperparams)