FROM nvidia/cuda:11.1-runtime AS pytorch-base
WORKDIR /
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    apt-utils \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade
RUN pip3 install numpy pandas jupyter pydantic tqdm click loguru
RUN pip3 install \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install \
    pytorch-lightning lightning-flash\
    fastai \
    kornia albumentations\
    tensorboardX \
    streamlit \
    seaborn \
    scikit-learn \
    wandb \
    data-science-types

FROM pytorch-base
RUN pip3 install \
    transformers \
    barbar \