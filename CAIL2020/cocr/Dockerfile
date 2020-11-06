FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

RUN mkdir -p /daas
WORKDIR /daas

COPY . /daas

RUN pip config set global.index-url https://pypi.douban.com/simple
RUN pip install -r requirements.txt -i https://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN rm -rf /root/.cache/pip

# Run when the container launches
CMD ["python", "one_for_all_inference.py"]