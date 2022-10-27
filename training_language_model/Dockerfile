# CPU Only, no support still for MPS and MacOs in general, going for remote NVIDIA workstation
FROM nvcr.io/nvidia/pytorch:22.07-py3

COPY transformer transformer
COPY datasets datasets
COPY test_GPU.py test_GPU.py
COPY main.py main.py


RUN python3 test_GPU.py

CMD [ "python3", "./main.py"]