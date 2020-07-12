
# The name of our algorithm
algorithm_name=coa-gbr
ls
account=$(aws sts get-caller-identity --query Account --output text)
echo $account
# Get the region defined in the current configuration (default to us-west-1 if none defined)
region=$(aws configure get region)
region=${region:-us-west-1}
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
echo $fullname
# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi
# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)
# Get the login command from ECR in order to pull down the SageMaker PyTorch image
$(aws ecr get-login --registry-ids 763104351884 --region ${region} --no-include-email)
# Build the docker image locally with the image name and then push it to ECR
# with the full name.
cd ~/coa_dev/coa/code/
docker build -t ${algorithm_name} -f sagemaker_helpers/Dockerfile . --build-arg REGION=${region}
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}