FROM ubuntu:22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# These get overwritten by values in compose.yaml
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12

# Install system dependencies & PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg2 \
    nano \
    git \
    gosu \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update

# Install Python version and build tools
RUN apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    coinor-libipopt-dev \
    gcc g++ gfortran git patch pkg-config \
    liblapack-dev libmetis-dev cmake vim \
    && rm -rf /var/lib/apt/lists/*

# Set Python version as the default 'python' and 'python3'
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install and upgrade pip for python version
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

RUN python -m pip install -U pip && python -m pip install -U "raddo[extras, cuda${CUDA_VERSION}]"

# Create a docker user with a default UID/GID (will be changed at runtime)
RUN groupadd -g 1000 docker && \
    useradd -u 1000 -g docker -m -s /bin/bash docker

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
