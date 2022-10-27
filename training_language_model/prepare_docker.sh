#!/bin/bash

# it works for Linux AMI 2 on AWS

# installing docker
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
apt-cache policy docker-ce
sudo apt install docker-ce -Y

# Configure Docker to start on boot
sudo systemctl enable docker.service

# Prepare Docker compose
sudo curl -L https://github.com/docker/compose/releases/download/1.21.0/docker-compose-`uname -s`-`uname -m` | sudo tee /usr/local/bin/docker-compose > /dev/null
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# Make ec2-user to docker group to run command without sudo
sudo usermod -aG docker ${USER}
