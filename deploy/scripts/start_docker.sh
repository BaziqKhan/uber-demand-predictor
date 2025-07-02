#!/bin/bash

cd /home/ubuntu/app

# === Load .env into environment ===
# this exports every key=value in .env
# if [ -f .env ]; then
#   set -o allexport
#   source .env
#   set +o allexport
# fi

# Login to ECR
aws ecr get-login-password --region eu-north-1 | \
docker login --username AWS --password-stdin 835952944694.dkr.ecr.eu-north-1.amazonaws.com

# Pull the latest image from ECR
docker pull 835952944694.dkr.ecr.eu-north-1.amazonaws.com/uber-demand:latest

if [ "$(docker ps -q -f name=uber-app)" ]; then
  docker stop uber-app
  docker rm uber-app
fi

docker run -d --name uber-app -p 80:8000 \
  -e DAGSHUB_USER_TOKEN=3356c1afc3e5674ab5682054248eda6706c50b5e \
  835952944694.dkr.ecr.eu-north-1.amazonaws.com/uber-demand:latest