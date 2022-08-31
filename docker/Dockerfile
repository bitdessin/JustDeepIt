 ARG base="11.6.2-cudnn8-devel-ubuntu20.04"
FROM nvidia/cuda:${base}

# cuda env
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# devel libraries and python packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    ffmpeg python3-dev python3-opencv ca-certificates git wget sudo ninja-build \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*  \
    && ln -svf /usr/bin/python3 /usr/bin/python \
    && wget https://bootstrap.pypa.io/pip/3.6/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py \
    && pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir tensorboard cmake \
    && pip install --no-cache-dir torch==1.11.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip install mmcv-full \
    && git clone https://github.com/open-mmlab/mmdetection.git mmdetection \
    && pip install --no-cache-dir -r ./mmdetection/requirements/build.txt \
    && pip install --no-cache-dir ./mmdetection \
    && rm -rf ./mmdetection \
    && pip install --no-cache-dir 'git+https://github.com/facebookresearch/fvcore' \
    && pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' \
    && pip install --no-cache-dir JustDeepIt

# user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser



