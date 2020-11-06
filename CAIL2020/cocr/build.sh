docker build -f Dockerfile -t pytorch:1.0 . --network=host
docker login https://hub.ai.xm.gov.cn
docker tag pytorch:1.0 hub.ai.xm.gov.cn/comp_2416/pytorch:1.0
docker push hub.ai.xm.gov.cn/comp_2416/pytorch:1.0