FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV PIP_NO_CACHE_DIR=1

ENV MATTERSIM_ROOT=/opt/Matterport3DSimulator
ENV OPENCV_ROOT=/opt/opencv-3.4.15
ENV OpenCV_DIR=/opt/opencv-3.4.15/build
ENV PYTHONPATH=${MATTERSIM_ROOT}/build
ENV LD_LIBRARY_PATH=${CONDA_DIR}/envs/navgpt2/lib:/usr/local/cuda/lib64
ENV HF_ENDPOINT=https://hf-mirror.com
ENV NAVGPT2_ASSET_ROOT=/workspace/vln-lab/data/navgpt2
ENV NAVGPT2_DATASETS=${NAVGPT2_ASSET_ROOT}/datasets
ENV NAVGPT2_QFORMER_DIR=${NAVGPT2_ASSET_ROOT}/map_nav_src/models/lavis/output

SHELL ["/bin/bash", "-lc"]

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git git-lfs unzip bzip2 ca-certificates bash \
    build-essential ninja-build pkg-config cmake doxygen \
    vim tmux \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1 libglvnd-dev libopengl-dev libegl1 libegl1-mesa-dev libgl1-mesa-dev \
    libjsoncpp-dev libepoxy-dev libglm-dev \
    libosmesa6 libosmesa6-dev libglew-dev \
    python3-dev python3-pip python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh && \
    conda clean -afy

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

RUN mkdir -p /root/.pip && cat > /root/.pip/pip.conf <<'EOF_PIP'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF_PIP

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

RUN conda create -n navgpt2 python=3.10 -y && \
    conda clean -afy

RUN conda run -n navgpt2 python -m pip install --upgrade pip setuptools wheel

COPY envs/navgpt2/requirements-cu128-py310.txt /tmp/navgpt2-requirements-cu128-py310.txt

# Needed by MatterSim CMake FindNumPy. Install the rest after manually
# installing the CUDA 12.8 torch/torchvision wheels inside the container.

# Install all remaining Python dependencies.
RUN conda run -n navgpt2 python -m pip install -r /tmp/navgpt2-requirements-cu128-py310.txt

# C++ OpenCV 3.4.15 for Matterport3DSimulator-revised compatibility.
ARG OPENCV_VERSION=3.4.15
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

# Compile MatterSim in the same ABI/Python environment as NavGPT-2.
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

WORKDIR /workspace/vln-lab

CMD ["/bin/bash"]
