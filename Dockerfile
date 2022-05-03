FROM python:3.6
WORKDIR /usr/src/app
COPY . .
RUN pip install tensorflow-gpu==2.1.0
RUN pip install tensorflow-serving-api-gpu==2.1.0
RUN pip install tensorboard==2.1.0
RUN pip install websockets
RUN pip install numpy
RUN pip install scikit-image
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
ENTRYPOINT [ "python", "snva.py", "--modelsdirpath", "/usr/model", "-op", "/usr/output", "-ip", "/usr/videos", "-l", "/usr/logs"]