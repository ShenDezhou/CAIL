# COCR

中文语句中的符号（-/+*#%@）,英文字符（&<>=[]{}）。

账号：comp_2416
密码：YAd0BOFsDW6wFU07 和 WllPeMX8WojBxkgl

yum install docker-ce-19.03.13 docker-ce-cli-19.03.13 containerd.io


# Stop running contrainers

docker stop $(docker ps -aq)
docker rm -vf $(docker ps -aq)

docker run --rm -v /pos_a:/daas/data pytorch:1.0 python one_for_all_inference.py -e data/

# which docker to use?

* FACEBOOK `pytorch1.5` is too old to use, it use ubuntu 16.04 and has many problems with python3-opencv.

* use NVIDIA latest version: `docker pull nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04`

* some tips: `docker commit [OPTIONS]  CONTAINER(容器名或容器ID)  [REPOSITORY[:TAG]](镜像名或镜像ID)`

# centos install gpu-docker tools

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

# install nvidia-container-toolkit

sudo yum install -y nvidia-container-toolkit
sudo systemctl restart docker

then you can run docker with GPU support. 
`docker run -it --gpus all --network host pytorch:1.1 bash`