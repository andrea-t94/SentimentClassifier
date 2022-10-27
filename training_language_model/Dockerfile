# CPU Only, no support still for MPS and MacOs in general, going for remote NVIDIA workstation
FROM nvcr.io/nvidia/pytorch:22.06-py3

COPY ./transformer transformer
COPY ./datasets datasets
COPY ./training_language_model .

RUN pip3 install -r training_language_model/requirements.txt
# TODO: doesn't make sense to print GPU enabled here because it works when to run
RUN python3 training_language_model/test_GPU.py

CMD [ "python3", "training_language_model/main.py"]
