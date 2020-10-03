FROM tensorflow/tensorflow:2.3.0-gpu-jupyter 

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt update && apt install -y libsm6 libxext6 libxrender1