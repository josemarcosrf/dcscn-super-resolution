FROM nvidia/cuda

SHELL ["/bin/bash", "-c"]

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    libopenblas-base libomp-dev \
    python3.6 python3-pip \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy as early as possible so we can cache ...
COPY requirements.txt .
RUN pip3 install -r requirements.txt --no-cache-dir

COPY ./ /super-resolution
WORKDIR /super-resolution

RUN mkdir /super-resolution/app

# volumes to link checkpoints, data and logs
VOLUME ["/super-resolution/app/checkpoints", \
        "/super-resolution/app/data", \
        "/super-resolution/app/logs"]

ENTRYPOINT ["./entrypoint.sh"]

CMD ["python", "train.py"]
