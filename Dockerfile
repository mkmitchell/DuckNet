# Use Ubuntu operating system
FROM ubuntu
ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"
WORKDIR /app
# Create /app folder on docker container
# Copy files from current folder to /app folder
COPY * /app
# Install Python, pip, conda
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda init bash \
    && . ~/.bashrc \
    && conda env create --file environment.yml \
    && conda activate DuckNet
RUN python fetch_pretrained_models.py
# By default Voilà uses port 8866
EXPOSE 5050
# Run voila when the cotainer is launched
CMD ["/root/miniconda/3/envs/DuckNet/bin/gunicorn", "main:app", "--bind 0.0.0.0:5050"]
# CMD bash