FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:ubuntugis/ppa \
    && apt-get update \
    && apt-get -y install --no-install-recommends \
    build-essential \
    ninja-build \
    libgdal-dev \
    gdal-bin \
    python3-dev \
    python3-pip \
    python3-opencv \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Python dependencies
# We install torch CPU version first to avoid pulling in huge CUDA deps
RUN python3 -m pip install --upgrade pip wheel \
    && python3 -m pip install opencv-python cython \
    && python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install wandb sphinx pycocotools

# Install detectron2
RUN python3 -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Install other dependencies from setup.py
RUN python3 -m pip install pyyaml numpy pandas tqdm shapely geopandas "rasterio<1.4" fiona rtree

# Clean up
RUN python3 -m pip cache purge
