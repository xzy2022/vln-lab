FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV MP3DSIM_CONDA_PREFIX=${CONDA_DIR}/envs/mp3d-sim
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV PIP_NO_CACHE_DIR=1
ENV MATTERSIM_ROOT=/workspace/vln-lab/third_party/Matterport3DSimulator
ENV PYTHONPATH=${MATTERSIM_ROOT}/build
ENV OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV LD_LIBRARY_PATH=${MP3DSIM_CONDA_PREFIX}/lib:/usr/local/cuda/lib64
ENV LIBRARY_PATH=${MP3DSIM_CONDA_PREFIX}/lib
ENV LDFLAGS="-L${MP3DSIM_CONDA_PREFIX}/lib -Wl,-rpath,${MP3DSIM_CONDA_PREFIX}/lib"

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git git-lfs bzip2 ca-certificates bash \
    build-essential ninja-build pkg-config cmake doxygen \
    vim tmux \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1 libglvnd-dev libopengl-dev libegl1 libegl1-mesa-dev libgl1-mesa-dev \
    libjsoncpp-dev libepoxy-dev libglm-dev \
    libosmesa6 libosmesa6-dev libglew-dev \
    libopencv-dev \
    python3-dev python3-pip python3-setuptools \
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

RUN mkdir -p /root/.pip && cat > /root/.pip/pip.conf <<'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF

RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate mp3d-sim" >> /root/.bashrc && \
    echo "export MATTERSIM_ROOT=/workspace/vln-lab/third_party/Matterport3DSimulator" >> /root/.bashrc && \
    echo 'export OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4' >> /root/.bashrc && \
    echo 'export CC=/usr/bin/gcc' >> /root/.bashrc && \
    echo 'export CXX=/usr/bin/g++' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> /root/.bashrc && \
    echo 'export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}' >> /root/.bashrc && \
    echo 'export LDFLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib"' >> /root/.bashrc && \
    echo 'export PYTHONPATH=${MATTERSIM_ROOT}/build:${PYTHONPATH:-}' >> /root/.bashrc

SHELL ["/bin/bash", "-lc"]

RUN conda create -n mp3d-sim python=3.10 -y && \
    conda install -n mp3d-sim -c conda-forge -y libstdcxx-ng libgcc-ng && \
    conda clean -afy

RUN conda run -n mp3d-sim python -m pip install --upgrade pip setuptools wheel && \
    conda run -n mp3d-sim python -m pip install \
        numpy opencv-python pillow imageio networkx matplotlib

WORKDIR /workspace/vln-lab

CMD ["/bin/bash"]
