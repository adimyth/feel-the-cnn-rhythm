docker build -t tensorboard:latest -f docker/tensorboard.Dockerfile .
docker build -t pytorch:latest -f docker/pytorch.Dockerfile .
docker build -t tensorflow:latest -f docker/tensorflow.Dockerfile .