ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${PYTORCH_IMAGE}
ENTRYPOINT []

WORKDIR /app/OpenVoice
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=20

RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
        /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        libavcodec-dev \
        libavdevice-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        libswscale-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY OpenVoice/requirements.txt /app/OpenVoice/requirements.txt
RUN pip install --no-cache-dir --no-deps "torchaudio==2.8.0" \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r /app/OpenVoice/requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY OpenVoice /app/OpenVoice
COPY data/input/audio/yongen.wav /app/data/input/audio/yongen.wav

EXPOSE 8084
HEALTHCHECK --interval=20s --timeout=5s --start-period=10m --retries=10 \
    CMD curl -fsS http://localhost:8084/health || exit 1

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "8084"]
