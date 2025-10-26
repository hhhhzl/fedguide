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

# --- Python 3.10 ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3.10-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 && \
    python3 -m pip install --no-cache-dir -U pip setuptools wheel

RUN apt-get update && apt-get install -y python-is-python3 libffi-dev python3-dev

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
COPY . /workspace/fedguide

RUN chmod +x scripts/*.sh || true

RUN /workspace/fedguide/scripts/install_mujoco_linux.sh
RUN /workspace/fedguide/scripts/install_mujoco_py_linux.sh
RUN /workspace/fedguide/scripts/setup.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]