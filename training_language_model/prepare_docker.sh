#!/bin/bash

# PLA   ESE NOTE: it works on installing Docker on Ubuntu 22.04. Not tested on other systems

sudo apt update
# dependencies
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common -y
# add docker official GPG key
# Since Ubuntu 22.04 is yet to be officially released, add the repository for Ubuntu 20.04 Stable.
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# then add their repo
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"

# finally install it
sudo apt install docker-ce docker-ce-cli containerd.io -y

# add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker