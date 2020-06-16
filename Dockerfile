# Copyright 2020 Marco Franchi
# SPDX-License-Identifier: MIT

FROM python:3.6.3
FROM tensorflow/tensorflow:latest-py3

RUN apt-get update -y
RUN apt-get install -y build-essential python3-pip python3-dev wget
RUN apt-get install -y python3-setuptools

RUN pip3 install --upgrade pip

RUN apt-get install -y gir1.2-gst-plugins-base-1.0 \
		gir1.2-gstreamer-1.0 \
		gir1.2-gtk-3.0

RUN apt-get install -y pkg-config \
		libgirepository1.0-dev \
		gcc \
		libcairo2-dev

RUN apt-get install -y gstreamer1.0-plugins-bad \
		gstreamer1.0-plugins-good \
		gstreamer1.0-plugins-ugly \
		gstreamer1.0-libav

RUN apt-get install -y python3-cairo \
		python3-gi \
		python3-gi-cairo \
		python3-gst-1.0 \
		python3-numpy \
		python3-pil \
		python3-dev

RUN pip3 install jupyter
RUN pip3 install pycairo PyGObject opencv-python

RUN wget https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl
RUN pip3 install tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl

RUN mkdir src
WORKDIR src/
COPY . .

WORKDIR /src

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
