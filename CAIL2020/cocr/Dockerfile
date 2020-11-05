FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

RUN mkdir -p /daas
WORKDIR /daas

COPY requirements-service.txt /daas
COPY . /daas

RUN pip install -r requirements-service.txt && rm -rf /root/.cache/pip

# Run when the container launches
CMD ["python", "deploy_models.py"]