FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

# 系统基础与包管理行为
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PIP_NO_CACHE_DIR=1

# Conda 安装位置；提前放入 PATH，后续 RUN 可以直接调用 conda。
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# VLN-MME 以源码方式运行，顶层 import 依赖 src 进入 Python 搜索路径。
ENV PYTHONPATH=/workspace/vln-lab/third_party/VLN-MME/src

SHELL ["/bin/bash", "-lc"]

# 使用国内 Ubuntu 镜像源，加快 apt 下载。
RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list

# 基础系统依赖：下载工具、常用 shell 工具、编译基础包和轻量运行库。
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git git-lfs unzip bzip2 ca-certificates bash \
    build-essential ninja-build pkg-config \
    vim tmux less \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# VLN-MME 当前不需要额外系统依赖；如后续模型需要系统库，在这里新增。

# 安装 Miniconda
RUN wget -qO /tmp/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh && \
    conda clean -afy

# 固定 conda 使用 defaults，并走清华镜像。
RUN cat > /root/.condarc <<'EOF_CONDA'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF_CONDA

# 固定 pip 使用阿里云镜像。
RUN mkdir -p /root/.pip && cat > /root/.pip/pip.conf <<'EOF_PIP'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF_PIP

# 创建 conda 环境
RUN conda create -n vlnmme python=3.10 -y && conda clean -afy
RUN conda run -n vlnmme python -m pip install --upgrade pip setuptools wheel

# 安装 requirements 依赖
COPY envs/vlnmme/requirements-cu128-py310.txt /tmp/vlnmme-requirements.txt
RUN conda run -n vlnmme python -m pip install -r /tmp/vlnmme-requirements.txt

# 配置环境变量
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate vlnmme" >> /root/.bashrc

WORKDIR /workspace/vln-lab

CMD ["/bin/bash"]
