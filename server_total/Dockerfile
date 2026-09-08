ARG REGISTRY=
FROM ${REGISTRY}python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN sed -i \
        -e 's|http://deb.debian.org/debian-security|https://mirrors.aliyun.com/debian-security|g' \
        -e 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY server_total/requirements.txt /app/server_total/requirements.txt
RUN pip install --no-cache-dir -r /app/server_total/requirements.txt

COPY llm_inference /app/llm_inference
COPY server_total /app/server_total

EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "server_total.app:app", "--host", "0.0.0.0", "--port", "8080"]
