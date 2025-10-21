FROM ubuntu:22.04

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    AUTO_INSTALL_CUDA_ON_GPU="1" \
    CUDA_VERSION="11-8" \
    CONDA_DIR="/opt/conda"
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# --- Essentials ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git \
      software-properties-common build-essential \
      pkg-config openssh-client \
      libc6 libgcc1 libstdc++6 \
      gnupg dirmngr \
      unzip \
    && rm -rf /var/lib/apt/lists/*

# --- Python 3.9 ---
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
         python3.9 python3.9-venv python3.9-dev python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.9 /tmp/get-pip.py \
    && rm /tmp/get-pip.py

RUN apt-get update && apt-get install -y python-is-python3 libffi-dev python3-dev \

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# # --- Miniconda  ---
# ARG MINICONDA_INSTALLER=Miniconda3-py39_24.1.2-0-Linux-x86_64.sh
# RUN curl -fsSL "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" -o /tmp/miniconda.sh \
#     && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
#     && rm -f /tmp/miniconda.sh \
#     && conda update -n base -c defaults -y conda \
#     && conda clean -afy

# --- CUDA runtime setup script (runs at container start if GPU present) ---
COPY /docker/setup-cuda.sh /opt/setup-cuda.sh
RUN chmod +x /opt/setup-cuda.sh

# Entrypoint wrapper: conditionally installs CUDA libs once, then launches the user command
COPY /docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace/fedguide
COPY .. /workspace/fedguide

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]