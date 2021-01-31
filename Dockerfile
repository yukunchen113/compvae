FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /code
COPY requirements.txt /code/requirements.txt
COPY docker_pyvista.sh /code/docker_pyvista.sh
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
RUN apt update && apt install -y libgl1-mesa-glx libusb-1.0-0 libgl1-mesa-dev xvfb && . docker_pyvista.sh
#CMD ["python","train.py"]