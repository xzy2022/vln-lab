FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV PIP_NO_CACHE_DIR=1

# Runtime data/cache paths and HF endpoint are supplied by
# scripts/setup/run_vlnmme_container.sh.
ENV PYTHONPATH=/workspace/vln-lab/third_party/VLN-MME/src

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      wget curl git git-lfs unzip bzip2 ca-certificates bash \
      build-essential ninja-build pkg-config \
      vim tmux less \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -afy

RUN cat > /root/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

RUN mkdir -p /root/.pip && cat > /root/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 120
EOF

SHELL ["/bin/bash", "-lc"]

RUN conda create -n vlnmme python=3.10 -y && conda clean -afy

RUN conda run -n vlnmme python -m pip install --upgrade pip setuptools wheel

# CUDA 12.8 PyTorch. Do not inherit VLN-MME's cu118 torch requirement.
# RUN conda run -n vlnmme python -m pip install torch==2.7.0 torchvision==0.22.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ -i https://mirrors.aliyun.com/pypi/simple/

COPY envs/vlnmme/requirements-cu128-py310.txt /tmp/vlnmme-requirements.txt

RUN conda run -n vlnmme python -m pip install -r /tmp/vlnmme-requirements.txt

RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate vlnmme" >> /root/.bashrc

WORKDIR /workspace/vln-lab

CMD ["/bin/bash"]
