# CPU Only, no support still for MPS and MacOs in general, going for remote NVIDIA workstation
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY params ./params
COPY utils ./utils
COPY requirements.txt ./requirements.txt
COPY main.py ./main.py
COPY test.py ./test.py


# this is the command docker run -v $(pwd):/workspace/data docker-ml-model
# TODO: publish the images to be used in EC2
# TODO: modify docker compose to be able to be used on EC2 with pulled images (change volume and images reference)
# TODO: test GPU on EC2
# TODO: develop script that send them back to my laptop
ARG DIRPATH
ARG HF_USER
ARG DATASET_VERSION
ARG MODEL_VERSION_MLM

RUN mkdir $DIRPATH
RUN mkdir $DIRPATH/$MODEL_VERSION_MLM
RUN mkdir $DIRPATH/$MODEL_VERSION_MLM/models
RUN mkdir $DIRPATH/$MODEL_VERSION_MLM/runs

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install git-lfs
RUN git lfs install                                                         
RUN git clone https://huggingface.co/datasets/$HF_USER/$DATASET_VERSION

#RUN pip3 install pandas
RUN pip3 install -r requirements.txt
CMD [ "python3", "./main.py"]

