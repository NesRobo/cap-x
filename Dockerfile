FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    UV_LINK_MODE=copy \
    # Adds /root/.local/bin to the executable search path while preserving the existing paths
    PATH="/root/.local/bin:$PATH"  

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    build-essential \
    pkg-config \
    cmake \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install Python versions for later use
RUN uv python install 3.10.22 3.12

WORKDIR /workspace
RUN git clone \
    --recurse-submodules \
    https://github.com/NesRobo/cap-x.git \
    cap-x
WORKDIR /workspace/cap-x

# 4. Copy CaP-X and checked-out submodules
# WORKDIR /workspace/cap-x
# COPY . .

# 5. Install Robosuite (Python 3.10)
ENV UV_PROJECT_ENVIRONMENT=/workspace/cap-x/.venv

RUN uv sync --frozen \
    --python 3.10.22 \
    --extra robosuite

# 6. Install LIBERO (Python 3.12) - In the capx original README Libero 
# requires a separate environment 
# install Robosuite first and make LIBERO installation
# optional until we resolve the project's Python-version constraint.

# ENV UV_PROJECT_ENVIRONMENT=/workspace/cap-x/.venv-libero
# RUN uv sync --locked \
#    --python 3.12 \
#    --extra libero \
#    --extra contactgraspnet
# RUN uv venv /opt/venvs/libero --python 3.12

# 7. Default to Robosuite environment
# Installierte Umgebung standardmäßig verwenden
ENV PATH="/workspace/cap-x/.venv/bin:$PATH"

CMD ["/bin/bash"]