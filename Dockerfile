FROM nvidia/cuda:10.2-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update; apt install -y python3 python3-pip python3-opencv
ADD ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN mkdir /root/shadow-detection
ADD ./*.py /root/shadow-detection/
ADD ./*.sh /root/shadow-detection/
WORKDIR /root/shadow-detection/
