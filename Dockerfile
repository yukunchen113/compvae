FROM tensorflow/tensorflow:latest-gpu-py3
RUN useradd -d /lustre/work/yukunchen -M -N -u 7762 yukunchen
ENV HOME /lustre/work/yukunchen
WORKDIR $HOME/code
COPY . .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
RUN apt update 
RUN apt install -y libgl1-mesa-glx libusb-1.0-0 libgl1-mesa-dev xvfb
RUN chmod -R 777 $HOME