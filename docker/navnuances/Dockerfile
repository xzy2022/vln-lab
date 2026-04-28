FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV PIP_NO_CACHE_DIR=1

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git git-lfs unzip bzip2 ca-certificates bash build-essential vim tmux \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh && \
    conda clean -afy

RUN cat > /root/.condarc <<'EOF'
channels:
  - conda-forge
show_channel_urls: true
channel_priority: strict
default_channels: []
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
allowlist_channels:
  - conda-forge
EOF

RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc

RUN mkdir -p /root/.pip && cat > /root/.pip/pip.conf <<'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF

SHELL ["/bin/bash", "-lc"]

RUN conda create -n navnuances-eval python=3.10 -y && \
    conda clean -afy

RUN conda run -n navnuances-eval python -m pip install --upgrade pip setuptools wheel && \
    conda run -n navnuances-eval python -m pip install numpy networkx dtw==1.4.0 fastdtw==0.3.4 "gdown>=5.2.0" tqdm

RUN echo "conda activate navnuances-eval" >> /root/.bashrc

WORKDIR /workspace/vln-lab
CMD ["/bin/bash"]
