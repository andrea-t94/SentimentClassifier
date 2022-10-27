#!/bin/bash

# install the drivers recommended
sudo apt install ubuntu-drivers-common -y
sudo ubuntu-drivers autoinstall

# reboot the system
sudo reboot