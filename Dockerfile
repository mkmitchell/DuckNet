# Use Ubuntu operating system
FROM ubuntu
ENV PATH="/root/miniconda3/envs/DuckNet/bin/:/root/miniconda3/bin:$PATH"
ENV DOCKERPATH="/app"
ENV INSTANCE_PATH="/app"
ENV ROOT_PATH="/app"
ARG PATH="/root/miniconda3/bin:$PATH"
ENV PYTHONPATH="/app"
WORKDIR /app
# Create /app folder on docker container
# Copy files from current folder to /app folder
COPY . /app
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
# RUN ln -s /root/miniconda3/envs/DuckNet/bin/gunicorn /usr/bin/gunicorn
# use port 5050
EXPOSE 5050
# Run gunicorn when the container is launched
# increase timeout to 120 seconds and use gevent for async workers
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--timeout", "120", "--worker-class", "gevent", "maingunicorn:app"]
# CMD bash