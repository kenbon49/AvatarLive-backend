# 使用 NVIDIA CUDA 基础镜像
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -U openmim
RUN mim install "mmcv==2.0.1"
RUN mim install "mmdet==3.1.0"
RUN mim install "mmpose==1.1.0"
# 升级 pip
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制 requirements.txt 并安装依赖
COPY requirements.txt /app/
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# The runtime image contains CUDA 11.8 NVRTC but omits the unversioned dev link
# that CuDNN opens when compiling convolution kernels.
RUN ln -sf libnvrtc.so.11.2 /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so \
    && ldconfig \
    && python3 -c "import ctypes; ctypes.CDLL('libnvrtc.so')"

# 复制整个项目
COPY . /app

# 创建必要的目录（仅用于生产环境，开发模式会被 volume 挂载覆盖）
# models 目录应该从外部挂载，不需要在镜像中创建
# core_data 子目录会在 server.py 启动时自动创建
RUN mkdir -p /app/core_data

# 下载模型权重（这一步可能需要 5-10 分钟）
# RUN python3 download_weights.py || echo "⚠️ Model download skipped or failed, you may need to download manually"

# 暴露端口
EXPOSE 8083

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8083/health || exit 1

# 启动服务
CMD ["python3", "-m", "accelerated.server", "--host", "0.0.0.0", "--port", "8083"]
