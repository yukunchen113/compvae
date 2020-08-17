FROM tensorflow/tensorflow:2.2.0-gpu
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3-opencv
COPY . /usr/src/app/
RUN pip install --upgrade pip
RUN python3 -m pip install -r /usr/src/app/requirements.txt
WORKDIR /usr/src/app/
CMD ["python3", "test.py"]
