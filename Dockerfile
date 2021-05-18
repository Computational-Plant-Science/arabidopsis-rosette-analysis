FROM ubuntu:20.04

LABEL maintainer="Suxing Liu, Wes Bonelli"

COPY . /opt/spg

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    python3-setuptools \
    python3-pip \
    python3-numexpr \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgtk2.0-dev

RUN pip3 install --upgrade pip && \
    pip3 install -e /opt/spg

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
