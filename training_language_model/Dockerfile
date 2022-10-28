# CPU Only, no support still for MPS and MacOs in general, going for remote NVIDIA workstation
FROM nvcr.io/nvidia/pytorch:22.05-py3

COPY ./transformer transformer
COPY ./datasets datasets
COPY ./training_language_model .

RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.11.0 torchtext==0.12.0 torchdata==0.3.0
# TODO: doesn't make sense to print GPU enabled here because it works when to run
RUN python3 test_GPU.py

CMD [ "python3", "./main.py"]
