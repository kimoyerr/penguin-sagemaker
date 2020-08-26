# penguin-sagemaker
Read my blog for more details on this project:

## Pre-requisites
1. Make sure to create an IAM role that has full acess to all Sagemaker resources as well as ECR resources. Specify this IAM role when launching an EC2 instance
2. Make sure to use Sagemaker AMI to start creating an instance. I use the pytorch_p36 conda environment
3. Install docker-compose on the pytorch_p36 environment using instructions from here: https://docs.docker.com/compose/install/. docker-compose is needed to run the docker images locally 

## Build-images:
Use scripts folder:
`./scripts/*training.sh`
`

## Debugging-docker containers
`docker run -ti penguin-xgb-training /bin/bash`
`docker stop $(docker ps -aq)`
`docker rm $(docker ps -aq)`

Build the local image and run the tests iteratively. Maybe a better solution with docker and Pycharm

## Gotchas
The local output path to save the model and output files has to be the full absolute path
