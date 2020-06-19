# use python 3.7 as base image

FROM ubuntu:16.04
FROM python:3.7


# install dependencies
COPY requirements.txt /

RUN pip install opencv-python
RUN pip install -r requirements.txt

COPY trail / trail /
COPY newtf / newtf /
COPY imagelabels.mat /
COPY newclassifier.py /

RUN chmod u+x newclassifier.py

#Run  when container is launced
CMD ./newclassifier.py
