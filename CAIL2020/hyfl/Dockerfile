FROM machinelearning:1.0
USER root

COPY ./requirements.txt /workspace/requirements.txt
RUN  pip3 install -r /workspace/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

RUN rm -rf /workspace/*
COPY . /workspace
RUN rm -rf /root/.cache/pip/wheels/*
# Run when the container launches
WORKDIR /workspace
CMD ["python3"]
