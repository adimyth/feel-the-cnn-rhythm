FROM nvidia/cuda:11.1-runtime AS tensorflow-base
WORKDIR /
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    apt-utils \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade
RUN pip3 install \
    jupyter \
    numpy pandas \
    tensorflow \
    tensorboardX \
    poetry \
    pydantic \
    streamlit \
    seaborn \
    scikit-learn \
    tqdm \
    click \
    wandb

FROM tensorflow-base
RUN pip3 install \
    transformers \
    barbar \