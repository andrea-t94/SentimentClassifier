# CPU Only, no support still for MPS and MacOs in general, going for remote NVIDIA workstation
FROM arm64v8/python:3.9.14-slim

COPY transformer transformer
COPY datasets datasets
COPY test_GPU.py test_GPU.py
COPY main.py main.py


# TODO: use poetry since for Linux pip seems to have more support

RUN pip3 install torch==1.11.0 torchtext==0.12.0 torchdata==0.3.0
RUN pip3 install  whylogs==1.1.7
RUN python3 test_GPU.py

CMD [ "python3", "./main.py"]