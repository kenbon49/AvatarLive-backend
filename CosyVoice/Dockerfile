FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/third_party/Matcha-TTS \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        libsox-dev \
        sox \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/cosyvoice-requirements.txt
# openai-whisper 20231117 imports pkg_resources, which was removed in newer setuptools.
# Keep build isolation for TensorRT and the remaining dependencies, which need it.
# NVIDIA is the sole index for TensorRT because PyPI mirrors carry installer stubs.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=wheels,target=/tmp/tensorrt-wheels,readonly \
    python -m pip install --index-url "${PIP_INDEX_URL}" --retries 12 --timeout 300 "setuptools<81" \
    && python -m pip install --index-url https://pypi.nvidia.com --retries 12 --timeout 300 \
        /tmp/tensorrt-wheels/tensorrt_cu12_libs-10.13.3.9-py2.py3-none-manylinux_2_28_x86_64.whl \
        "tensorrt-cu12==10.13.3.9" \
        "tensorrt-cu12-bindings==10.13.3.9" \
    && python -m pip install --index-url "${PIP_INDEX_URL}" --no-build-isolation \
        --retries 12 --timeout 300 "openai-whisper==20231117" \
    && python -m pip install --index-url "${PIP_INDEX_URL}" --retries 12 --timeout 300 \
        -r /tmp/cosyvoice-requirements.txt

COPY . /app

EXPOSE 8084

CMD ["python", "server_v2.py", "--host", "0.0.0.0", "--port", "8084"]
