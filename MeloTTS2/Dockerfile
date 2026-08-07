FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
# Keep PyTorch, ONNX Runtime, CUDA, and cuDNN on one tested compatibility stack.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

# The PyTorch image ships matching CUDA 11.8 NVRTC under /opt/conda/lib but
# omits the unversioned link expected by CuDNN's runtime compiler.
RUN ln -sf libnvrtc.so.11.2 /opt/conda/lib/libnvrtc.so \
    && ldconfig \
    && python -c "import ctypes; ctypes.CDLL('libnvrtc.so')"

COPY . /app

EXPOSE 8084
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8084/health | grep -q '"gpu_ready":true' || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8084"]
