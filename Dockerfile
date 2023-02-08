#FROM python:3.8
ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

RUN pip install pydicom
RUN pip install pandas
RUN pip install tensorboard
RUN pip install opencv-python
#RUN pip install torch torchvision torchaudio
#RUN pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
RUN pip install monai==0.7.0
RUN pip install pytorch-ignite
RUN python -m pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg pylibjpeg-rle
RUN pip install tqdm

ADD inference.py .
ADD data_preparation.py .

RUN rm /opt/conda/lib/python3.7/ensurepip/_bundled/pip-20.1.1-py2.py3-none-any.whl
RUN rm /opt/conda/lib/python3.7/ensurepip/_bundled/setuptools-47.1.0-py3-none-any.whl
RUN rm /opt/conda/pkgs/conda-4.10.1-py37h06a4308_1/info/test/tests/conda_env/support/requirements.txt
RUN rm /opt/conda/pkgs/python-3.7.10-hdb3f193_0/lib/python3.7/ensurepip/_bundled/*.whl

ENTRYPOINT ["python3", "inference.py"]
