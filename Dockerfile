FROM jupyter/base-notebook:2023-03-13
ARG DEBIAN_FRONTEND=noninteractive
USER root
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install torch

COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /workdir/
