FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl software-properties-common \
	  && add-apt-repository -y ppa:deadsnakes/ppa \
	  && apt-get update \
	  && apt-get install -y python3.6
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6
RUN ln -s /usr/bin/python3.6 /usr/bin/python

RUN apt-get install -y --no-install-recommends \
    python3.6-dev git wget apt-utils vim openssh-server rabbitmq-server libmysqlclient-dev libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
	&& DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata \
	&& sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config \
	&& rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip
RUN pip install setuptools

RUN pip -V
RUN python -V

WORKDIR /workspace
ADD . .
RUN pip install torch==1.0.1.post2
RUN pip install tensorflow-gpu==1.10.0
RUN pip install opencv-python
RUN pip install keras==2.0.8
RUN pip install -r requirements.txt

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

RUN chmod -R a+w /workspace

EXPOSE 8000

#RUN /bin/bash