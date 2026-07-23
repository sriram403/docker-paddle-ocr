FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG UBUNTU_MIRROR=http://archive.ubuntu.com/ubuntu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt /tmp/requirements-api.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
       torch==2.10.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install \
       paddlepaddle-gpu==3.2.2 \
       --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ \
    && python3 -m pip install -r /tmp/requirements-api.txt

RUN sed -i "s|http://security.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install transformers==4.57.6

COPY . /app

RUN mkdir -p /outputs /logs /checkpoints

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
