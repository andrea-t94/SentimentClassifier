import platform
import torch

# TODO:
#  deploy on Ubuntu full code looking at Roboflow post
#  check if we can go for another pytorch version with MPS enabled for my local environment
#  develop with VS code container
#  test that training works --> make it train
#  refactor code and push on GitHub

if torch.cuda.is_available():
    print(f"CUDA version {(torch.version.cuda)} is available with {torch.cuda.device_count()} devices")
    if torch.backends.cudnn.version():
        print(f"cuDNN version {torch.backends.cudnn.version()} is also available")
    else:
        print(f"no cuDNN")
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    print("MPS is available.")


## STEPS:
## 1 TAKE LINUX AMI 2 WITH CUDA DRIVERS INSTALLED (OR UBUNTI AND FOLLOW THIS GUIDE https://www.linuxcapable.com/how-to-install-nvidia-drivers-on-ubuntu-22-04-lts/)
## 2 nvidia-smi to check NVIDIA version
## 3 sudo yum install python3
## 4 sudo yum install pip
## 5 pip3 install torch (from pytorch installation on Linux)
## 6 run the test and check CUDA version to use in docker
## 7 prepare docker (remembr sudo chmod u+x prepare_docker.sh )