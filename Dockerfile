FROM python:3.6
WORKDIR /usr/src/app
COPY . .
RUN pip install tensorflow==2.1.0
RUN pip install tensorboard==2.1.0
RUN pip install websockets
RUN pip install numpy
RUN pip install scikit-image
CMD [ "python", "snva.py", "-cnh", "localhost:8081", "-et", "-l", "./logs", "--modelname", "mobilenet_v2", "--modelsdirpath", "../models/work_zone_scene_detection/", "-msh", "0.0.0.0:8500"]