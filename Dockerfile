# Use NVIDIA CUDA 12.4.1 with cuDNN base image for GPU support
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PATH="/root/miniconda3/envs/DuckNet/bin/:/root/miniconda3/bin:$PATH"
ENV INSTANCE_PATH="/app"
ENV ROOT_PATH="/app"
ENV CONFIG_PATH="/app/settings.json"
ARG PATH="/root/miniconda3/bin:$PATH"
ENV PYTHONPATH="/app"

# Set the working directory in the container
WORKDIR /app

# Copy files from current folder to /app folder
COPY . /app

# Install Python, pip, conda, and git
RUN apt-get update
RUN apt-get install -y wget zip unzip git && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Initialize conda and create the environment
RUN conda init bash \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && . ~/.bashrc \
    && conda env create --file environment.yml \
    && conda activate DuckNet

# Install pt_soft_nms separately; throws errors if installed concurrently with torch
RUN /root/miniconda3/envs/DuckNet/bin/pip install --use-pep517 --no-build-isolation git+https://github.com/MrParosk/soft_nms.git

# Create necessary directories and zip models
RUN mkdir -p /app/models/detection
WORKDIR /app/models_src/2024-10-11/
RUN zip -r /app/models/detection/basemodel.pt.zip *

# Set the working directory back to /app
WORKDIR /app

# Expose port 5050
EXPOSE 5050

# Run waitress when the container starts
CMD ["python", "-u", "mainwaitress.py"]