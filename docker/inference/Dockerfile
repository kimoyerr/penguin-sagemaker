
# Build everything from scratch: Python 3
# From: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/multi_model_bring_your_own/container/Dockerfile
# With changes inspired from: https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/docker/1.5.0/py3/Dockerfile.cpu

FROM ubuntu:16.04

## Set a docker label to advertise multi-model support on the container. Turn off if needed
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Setup Environment for MME. Really crucial or will result in errors when adding modules
ENV SAGEMAKER_MULTI_MODEL=true
ENV SAGEMAKER_BIND_TO_PORT=${SAGEMAKER_BIND_TO_PORT:-8080}

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    curl \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
RUN pip3 --no-cache-dir install mxnet \
                                multi-model-server \
                                sagemaker-inference \
                                sklearn \
                                numpy \
                                pandas \
                                patsy \
                                joblib==0.14.1 \
                                matplotlib \
                                retrying

WORKDIR /

# Copy entrypoint script to the image
COPY sagemaker_helpers/coa_inference/container/mms-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/
RUN mkdir -p /opt/ml/model
RUN chmod +rwx /opt/ml/model

# Copy the default custom service file to handle incoming data and inference requests
COPY sagemaker_helpers/coa_inference/container/model_handler.py /home/model-server/model_handler.py

# Copy some dummy models to test. You can delete these later
#RUN mkdir dummy_1
#RUN chmod +rwx dummy_1
#COPY test/resources/models/dummy_1 dummy_1
#RUN mkdir dummy_2
#RUN chmod +rwx dummy_2
#COPY test/resources/models/dummy_1 dummy_2

# Copy the COA algorithm and data files
COPY learn /home/model-server/learn
COPY test/resources /home/model-server/resources

# Define an entrypoint script for the docker image
ENV SAGEMAKER_HANDLER="/home/model-server/model_handler.py:handle"
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]

