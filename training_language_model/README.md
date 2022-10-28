document the scope of this step and how does it work (using Ubuntu remote server)

step zero: prepare Ubuntu image
- docker preparation
- NVIDIA drivers for Ubuntu and Docker
- Ec2 with GPU + S3 access

first step: build docker image
docker build -f training_language_model/Dockerfile -t pytorch-gpu .

second step: run docker image enabling GPUs
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it pytorch-gpu
