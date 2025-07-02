#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

# Update and install Docker
sudo apt-get update -y
sudo apt-get install docker.io -y

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Install unzip and curl
sudo apt-get install unzip curl -y

# Download and install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/
sudo /home/ubuntu/aws/install

# Add ubuntu user to the docker group (so you can run docker without sudo)
sudo usermod -aG docker ubuntu

# Clean up installer files
rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws