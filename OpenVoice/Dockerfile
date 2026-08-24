FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app/OpenVoice
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY OpenVoice/requirements.txt /app/OpenVoice/requirements.txt
RUN pip install --no-cache-dir -r /app/OpenVoice/requirements.txt

COPY OpenVoice /app/OpenVoice
COPY data/input/audio/yongen.wav /app/data/input/audio/yongen.wav

EXPOSE 8084
HEALTHCHECK --interval=20s --timeout=5s --start-period=10m --retries=10 \
    CMD curl -fsS http://localhost:8084/health || exit 1

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8084"]
