mount -t xfs /dev/nvme0n1p1 /mnt/data
service docker restart
docker start $(docker ps -aq)
cd /mnt/data/unifyuix
nohup python torch_server.py &