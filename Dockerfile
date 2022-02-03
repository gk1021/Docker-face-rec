FROM ubuntu

RUN apt-get update
RUN apt-get -y install python3-pip

RUN apt install -y libprotobuf-dev protobuf-compiler

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y  cmake
RUN apt-get install -y libopenblas-dev liblapack-dev 
RUN apt-get install -y libx11-dev libgtk-3-dev
RUN apt-get install -y python3 python3-dev python3-pip

RUN pip3 install numpy pandas face_recognition Scikit-learn dlib docopt
RUN apt-get -y install python3-opencv

COPY Allface ./Allface
COPY Faces ./Faces
COPY Frames ./Frames
COPY video ./video
COPY face_rec.py ./face_rec.py

RUN bash -c "echo python3 face_rec.py 0 - for getting direct video from webcame or python3 face_rec.py 1 from presaved" 