FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Beijing /etc/localtime

RUN mkdir -p /root/.pip
RUN  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
COPY ./requirements.txt /workspace/requirements.txt
RUN  pip install -r /workspace/requirements.txt

RUN sed -i 's/archive.ubuntu/mirrors.aliyun/g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get install python3-opencv -y

COPY . /workspace
# Run when the container launches
CMD 'sh'
#ENTRYPOINT ['/workspace/one_for_all_inference.py']