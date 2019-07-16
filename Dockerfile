FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&  apt-get install -y   \
    python3-pip python3 \
    htop unzip wget sudo vim

RUN ln -s /usr/bin/python3  /usr/bin/python
RUN pip3 install pip --upgrade
# jupyter notebook and tensorboard
EXPOSE 8888 6006

COPY requirements.txt ./


RUN pip3 install --no-cache-dir -r requirements.txt


ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

ENV  PATH=/usr/local/nvidia/bin:$PATH
