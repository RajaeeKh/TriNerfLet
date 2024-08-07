# Reference:
# https://github.com/cvpaperchallenge/Ascender
# https://github.com/nerfstudio-project/nerfstudio

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04


# Set compute capability for nerfacc and tiny-cuda-nn
# See https://developer.nvidia.com/cuda-gpus and limit number to speed-up build
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV TCNN_CUDA_ARCHITECTURES=90;89;86;80;75;70;61;60
# Speed-up build for RTX 30xx
# ENV TORCH_CUDA_ARCH_LIST="8.6"
# ENV TCNN_CUDA_ARCHITECTURES=86
# Speed-up build for RTX 40xx
# ENV TORCH_CUDA_ARCH_LIST="8.9"
# ENV TCNN_CUDA_ARCHITECTURES=89

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/home/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

RUN apt-get update -y && apt-get upgrade -y

# apt install by root user
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    wget \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip setuptools==68.2.2 ninja
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# Install nerfacc and tiny-cuda-nn before installing requirements.txt
# because these two installations are time consuming and error prone
RUN pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

COPY requirements2.txt /tmp
RUN cd /tmp && pip install -r requirements2.txt



WORKDIR /home

RUN mkdir aux_projects
COPY aux_libs/ffmlp aux_projects/ffmlp
COPY aux_libs/freqencoder aux_projects/freqencoder
COPY aux_libs/gridencoder aux_projects/gridencoder
COPY aux_libs/raymarching aux_projects/raymarching
COPY aux_libs/shencoder aux_projects/shencoder
RUN mkdir aux_projects/scripts
COPY aux_libs/scripts/install_ext.sh aux_projects/scripts/install_ext.sh
WORKDIR /home/aux_projects
RUN bash scripts/install_ext.sh

WORKDIR /home
ENTRYPOINT sleep infinity



