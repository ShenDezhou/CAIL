FROM machinelearning:1.0
USER root

COPY ./requirements.txt /workspace/requirements.txt
RUN  pip3 install -r /workspace/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

RUN rm -rf /workspace/*
COPY . /workspace
#RUN pip3 install /workspace/torch-1.5.0-cp36-cp36m-manylinux1_x86_64.whl
RUN rm -rf /root/.cache/pip/wheels/*
# Run when the container launches
WORKDIR /workspace
CMD ["python3"]
#ENTRYPOINT ['/workspace/one_for_all_inference.py']