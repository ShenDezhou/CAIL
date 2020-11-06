FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

RUN mkdir -p /daas
WORKDIR /daas
COPY . /daas

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install python3.6
RUN ln -s /usr/bin/python3.6 /usr/bin/python
RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install -r requirements.txt -i https://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html




# Run when the container launches
CMD ["python", "one_for_all_inference.py"]