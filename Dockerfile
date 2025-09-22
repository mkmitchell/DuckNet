# Use NVIDIA CUDA 12.4.1 with cuDNN base image for GPU support
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/envs/DuckNet/bin/:/root/miniconda3/bin:$PATH"
ENV INSTANCE_PATH="/app"
ENV ROOT_PATH="/app"
ENV CONFIG_PATH="/app/settings.json"
ENV PYTHONPATH="/app"

WORKDIR /app

# Install everything in one layer and clean up thoroughly
RUN apt-get update && \
    apt-get install -y wget zip unzip git build-essential && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

# Copy environment file first for better caching
COPY environment.yml /app/

# Create conda environment and clean up
RUN conda init bash && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    . ~/.bashrc && \
    conda env create --file environment.yml && \
    conda clean -afy

RUN /root/miniconda3/envs/DuckNet/bin/pip install --use-pep517 --no-build-isolation git+https://github.com/MrParosk/soft_nms.git && \
    /root/miniconda3/envs/DuckNet/bin/pip cache purge

# Copy app code last
COPY . /app

RUN mkdir -p /app/models/detection && \
    cd /app/models_src/2024-10-11/ && \
    zip -r /app/models/detection/basemodel.pt.zip *

EXPOSE 5050
CMD ["python", "-u", "mainwaitress.py"]