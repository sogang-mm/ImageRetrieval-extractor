FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
	   git wget python-pip apt-utils vim openssh-server rabbitmq-server libmysqlclient-dev \
	&& DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata \
	&& sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config \
	&& rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip
RUN pip install setuptools

WORKDIR /workspace
ADD . .
RUN pip install -r requirements.txt

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

RUN chmod -R a+w /workspace

EXPOSE 8000

#RUN /bin/bash