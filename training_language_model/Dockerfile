# Since there is no support for MacOS systems as well as for MPS in general, Docker is useless on my current setup for training

FROM arm64v8/python:3.9.14-slim

COPY transformer transformer
COPY datasets datasets
COPY requirements.txt requirements.txt
COPY test.py test.py


# TODO: use poetry since for Linux pip seems to have more support
# CPU Only, no support still for MPS and MacOs in general
RUN pip3 install torch==1.11.0 torchtext==0.12.0 torchdata==0.3.0
RUN pip3 install  whylogs==1.1.7

# TODO: make it multi-arch and enable CUDA GPUs support (on X86 arch)
# When CUDA 11.6


CMD [ "python3", "./test.py"]