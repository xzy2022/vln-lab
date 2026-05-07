FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

# OpenCV 版本只在这里声明一次；OPENCV_ROOT、OpenCV_DIR 和下载地址都从它派生。
ARG OPENCV_VERSION=3.4.15

# 系统基础与包管理行为
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PIP_NO_CACHE_DIR=1

# Conda 安装位置；提前放入 PATH，后续 RUN 可以直接调用 conda。
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# VLN-Lab 统一约定：工作目录、数据根目录，以及国内 Hugging Face 镜像。
ENV HF_ENDPOINT=https://hf-mirror.com
ENV VLN_LAB_ROOT=/workspace/vln-lab
ENV VLN_LAB_DATA_ROOT=${VLN_LAB_ROOT}/data

# NavGPT-2 数据与 checkpoint 布局；运行脚本会读取这些路径。
ENV NAVGPT2_ASSET_ROOT=${VLN_LAB_DATA_ROOT}/navgpt2
ENV NAVGPT2_DATASETS=${NAVGPT2_ASSET_ROOT}/datasets
ENV NAVGPT2_QFORMER_DIR=${NAVGPT2_ASSET_ROOT}/map_nav_src/models/lavis/output

# NavGPT-2 运行依赖：MatterSim、OpenCV 和动态库搜索路径。
ENV MATTERSIM_ROOT=/opt/Matterport3DSimulator
ENV OPENCV_ROOT=/opt/opencv-${OPENCV_VERSION}
ENV OpenCV_DIR=${OPENCV_ROOT}/build
ENV PYTHONPATH=${MATTERSIM_ROOT}/build
ENV LD_LIBRARY_PATH=${CONDA_DIR}/envs/navgpt2/lib:/usr/local/cuda/lib64

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

# NavGPT-2/MatterSim 专属系统依赖：OpenCV/MatterSim 编译和 EGL/OSMesa 渲染。
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake doxygen \
    libglvnd-dev libopengl-dev libegl1 libegl1-mesa-dev libgl1-mesa-dev \
    libjsoncpp-dev libepoxy-dev libglm-dev \
    libosmesa6 libosmesa6-dev libglew-dev \
    python3-dev python3-pip python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -qO /tmp/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh && \
    conda clean -afy

# 固定 conda 使用 conda-forge，并走清华镜像。
RUN cat > /root/.condarc <<'EOF_CONDA'
channels:
  - conda-forge
show_channel_urls: true
channel_priority: strict
default_channels: []
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
allowlist_channels:
  - conda-forge
EOF_CONDA

# 固定 pip 使用阿里云镜像。
RUN mkdir -p /root/.pip && cat > /root/.pip/pip.conf <<'EOF_PIP'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF_PIP

# 创建 conda 环境
RUN conda create -n navgpt2 python=3.10 -y && \
    conda clean -afy
RUN conda run -n navgpt2 python -m pip install --upgrade pip setuptools wheel

# 安装 requirements 依赖
COPY envs/navgpt2/requirements-cu128-py310.txt /tmp/navgpt2-requirements-cu128-py310.txt
RUN conda run -n navgpt2 python -m pip install -r /tmp/navgpt2-requirements-cu128-py310.txt

# 编译 C++ OpenCV 3.x，保持与 Matterport3DSimulator-revised 的 ABI/接口兼容。
RUN cd /opt && \
    wget -q -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip -q opencv.zip && \
    rm opencv.zip && \
    cd opencv-${OPENCV_VERSION} && \
    mkdir -p build && cd build && \
    cmake \
      -D CMAKE_BUILD_TYPE=Release \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -D WITH_CUDA=OFF \
      .. && \
    cmake --build . -j"$(nproc)"

# 在 navgpt2 conda 环境中编译 MatterSim，确保 Python 绑定和运行环境一致。
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda activate navgpt2 && \
    git clone --recursive https://github.com/GengzeZhou/Matterport3DSimulator-revised.git ${MATTERSIM_ROOT} && \
    cd ${MATTERSIM_ROOT} && \
    git checkout opencv3.x-ubuntu20.04+ && \
    mkdir -p build && cd build && \
    cmake \
      -DEGL_RENDERING=ON \
      -DPYTHON_EXECUTABLE=${CONDA_DIR}/envs/navgpt2/bin/python \
      .. && \
    make -j"$(nproc)"

# 配置环境变量
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate navgpt2" >> /root/.bashrc && \
    echo "export MATTERSIM_ROOT=${MATTERSIM_ROOT}" >> /root/.bashrc && \
    echo "export PYTHONPATH=${MATTERSIM_ROOT}/build:\${PYTHONPATH:-}" >> /root/.bashrc && \
    echo "export OpenCV_DIR=${OpenCV_DIR}" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=${CONDA_DIR}/envs/navgpt2/lib:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}" >> /root/.bashrc && \
    echo "export NAVGPT2_ASSET_ROOT=\${NAVGPT2_ASSET_ROOT:-${NAVGPT2_ASSET_ROOT}}" >> /root/.bashrc && \
    echo "export NAVGPT2_DATASETS=\${NAVGPT2_DATASETS:-\${NAVGPT2_ASSET_ROOT}/datasets}" >> /root/.bashrc && \
    echo "export NAVGPT2_QFORMER_DIR=\${NAVGPT2_QFORMER_DIR:-\${NAVGPT2_ASSET_ROOT}/map_nav_src/models/lavis/output}" >> /root/.bashrc

RUN mkdir -p ${NAVGPT2_ASSET_ROOT}

WORKDIR /workspace/vln-lab

CMD ["/bin/bash"]
