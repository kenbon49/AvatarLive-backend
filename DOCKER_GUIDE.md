# Docker Compose 完整使用指南

## 目录
- [基础概念](#基础概念)
- [配置文件详解](#配置文件详解)
- [常用命令](#常用命令)
- [实际使用场景](#实际使用场景)
- [高级功能](#高级功能)
- [故障排查](#故障排查)

---

## 基础概念

### 什么是 Docker Compose？

Docker Compose 是一个用于**定义和运行多容器 Docker 应用**的工具。使用 YAML 文件配置应用的服务，然后通过一条命令就能启动所有服务。

### 核心优势

| 传统 Docker | Docker Compose |
|------------|----------------|
| `docker run` 命令长且复杂 | 配置写在 YAML 文件 |
| 多个容器需要多条命令 | 一条命令启动所有服务 |
| 网络需要手动创建 | 自动创建网络 |
| 难以版本控制 | 配置文件可版本控制 |

---

## 配置文件详解

### 文件结构

```yaml
version: '3.8'           # Compose 文件格式版本

services:                # 定义服务（容器）
  service_name:
    # 服务配置...

networks:                # 定义网络
  network_name:
    # 网络配置...

volumes:                 # 定义卷（可选）
  volume_name:
    # 卷配置...
```

### 关键配置项解析

#### 1. build（构建配置）

```yaml
build:
  context: .                    # 构建上下文（Dockerfile 所在目录）
  dockerfile: Dockerfile        # Dockerfile 文件名
  args:                         # 构建参数
    - HTTP_PROXY=${HTTP_PROXY}
```

等价于：
```bash
docker build --build-arg HTTP_PROXY=$HTTP_PROXY -f Dockerfile .
```

#### 2. image（镜像）

```yaml
image: musetalk:latest  # 镜像名:标签
```

- 如果有 `build`，构建后打上这个标签
- 如果没有 `build`，直接拉取这个镜像

#### 3. container_name（容器名）

```yaml
container_name: musetalk_server
```

不指定时，Docker 自动生成名称：`项目名_服务名_序号`（如 `musetalk_musetalk_1`）

#### 4. ports（端口映射）

```yaml
ports:
  - "8000:8000"      # 宿主机:容器
  - "8001-8003:8001-8003"  # 范围映射
```

等价于：
```bash
docker run -p 8000:8000
```

#### 5. volumes（卷挂载）

```yaml
volumes:
  # 绑定挂载（Bind Mount）
  - ./models:/app/models:ro     # 宿主机:容器:权限
  
  # 命名卷（Named Volume）
  - data_volume:/app/data
  
  # 匿名卷（Anonymous Volume）
  - /app/temp
```

**权限标志**：
- `ro`：只读（read-only）
- `rw`：读写（默认）

#### 6. environment（环境变量）

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0    # 键值对
  - DEBUG                     # 从宿主机继承
  
# 或者
environment:
  CUDA_VISIBLE_DEVICES: 0
  DEBUG: 1
```

#### 7. deploy（资源限制 - GPU 配置）

```yaml
deploy:
  resources:
    reservations:              # 预留资源
      devices:
        - driver: nvidia
          count: 1             # GPU 数量（1 或 all）
          capabilities: [gpu]
    limits:                    # 资源上限
      cpus: '2'
      memory: 8G
```

#### 8. restart（重启策略）

```yaml
restart: unless-stopped
```

| 策略 | 说明 |
|------|------|
| `no` | 不自动重启（默认） |
| `always` | 总是重启 |
| `on-failure` | 失败时重启 |
| `unless-stopped` | 除非手动停止，否则重启 |

#### 9. healthcheck（健康检查）

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/docs"]
  interval: 30s      # 检查间隔
  timeout: 10s       # 超时时间
  retries: 3         # 重试次数
  start_period: 60s  # 启动宽限期
```

#### 10. depends_on（依赖关系）

```yaml
depends_on:
  - database         # 先启动 database，再启动当前服务
  - redis
```

**注意**：只保证启动顺序，不等待服务就绪！

#### 11. networks（网络）

```yaml
networks:
  - musetalk_network
```

---

## 常用命令

### 基础操作

```bash
# 1. 启动所有服务（前台运行）
docker-compose up

# 2. 启动所有服务（后台运行）
docker-compose up -d

# 3. 停止所有服务
docker-compose down

# 4. 停止并删除所有资源（容器、网络、卷）
docker-compose down -v

# 5. 重启服务
docker-compose restart

# 6. 查看服务状态
docker-compose ps

# 7. 查看日志
docker-compose logs -f          # 所有服务
docker-compose logs -f musetalk # 指定服务
```

### 构建与更新

```bash
# 构建镜像
docker-compose build

# 强制重新构建（不使用缓存）
docker-compose build --no-cache

# 构建并启动
docker-compose up --build

# 拉取最新镜像
docker-compose pull
```

### 服务管理

```bash
# 启动指定服务
docker-compose up musetalk

# 停止指定服务
docker-compose stop musetalk

# 删除指定服务
docker-compose rm musetalk

# 扩展服务实例（负载均衡）
docker-compose up -d --scale musetalk=3
```

### 调试与维护

```bash
# 进入容器终端
docker-compose exec musetalk bash

# 在容器中执行命令
docker-compose exec musetalk python3 --version

# 查看资源占用
docker-compose top

# 验证配置文件
docker-compose config

# 查看镜像
docker-compose images
```

---

## 实际使用场景

### 场景 1：首次启动项目

```bash
# 1. 克隆项目
cd D:\code\avatar\MuseTalk

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 3. 构建并启动
docker-compose up -d

# 4. 查看日志
docker-compose logs -f

# 5. 检查健康状态
docker-compose ps
```

### 场景 2：开发模式（代码热更新）

```bash
# 使用开发配置
docker-compose -f docker-compose.dev.yml up

# 代码修改后自动生效（因为挂载了源码目录）
# 如果需要重启：
docker-compose -f docker-compose.dev.yml restart
```

### 场景 3：生产部署

```bash
# 使用生产配置
docker-compose -f docker-compose.prod.yml up -d

# 查看 Nginx 日志
docker-compose -f docker-compose.prod.yml logs -f nginx

# 滚动更新（无停机）
docker-compose -f docker-compose.prod.yml up -d --no-deps --build musetalk
```

### 场景 4：多环境管理

```bash
# 基础配置 + 开发覆盖
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# 基础配置 + 生产覆盖
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 场景 5：数据备份与恢复

```bash
# 备份数据卷
docker-compose exec musetalk tar czf /tmp/backup.tar.gz /app/core_data
docker cp $(docker-compose ps -q musetalk):/tmp/backup.tar.gz ./backup.tar.gz

# 恢复数据
docker cp backup.tar.gz $(docker-compose ps -q musetalk):/tmp/
docker-compose exec musetalk tar xzf /tmp/backup.tar.gz -C /
```

---

## 高级功能

### 1. 环境变量替换

**docker-compose.yml**：
```yaml
services:
  musetalk:
    image: musetalk:${VERSION:-latest}  # 默认 latest
    ports:
      - "${API_PORT:-8000}:8000"
```

**.env**：
```bash
VERSION=v1.5
API_PORT=9000
```

### 2. 扩展与覆盖

**docker-compose.yml**（基础）：
```yaml
services:
  musetalk:
    image: musetalk:latest
    ports:
      - "8000:8000"
```

**docker-compose.override.yml**（自动合并）：
```yaml
services:
  musetalk:
    environment:
      - DEBUG=1
```

启动时自动合并两个文件。

### 3. 配置锚点（YAML Anchors）

```yaml
# 定义可复用配置
x-common-config: &common
  restart: unless-stopped
  logging:
    driver: json-file
    options:
      max-size: "100m"

services:
  service1:
    <<: *common  # 引用
    image: app1:latest
  
  service2:
    <<: *common
    image: app2:latest
```

### 4. 依赖等待（使用 wait-for-it）

```yaml
services:
  database:
    image: postgres:14
  
  app:
    depends_on:
      - database
    command: >
      bash -c "
        ./wait-for-it.sh database:5432 --timeout=60 --strict &&
        python3 server.py
      "
```

### 5. 多阶段构建优化

**Dockerfile**：
```dockerfile
# 构建阶段
FROM python:3.10 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# 运行阶段（更小）
FROM python:3.10-slim
COPY --from=builder /install /usr/local
COPY . /app
CMD ["python3", "server.py"]
```

---

## 故障排查

### 常见问题

#### 1. GPU 不可用

```bash
# 检查 nvidia-docker 运行时
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# 如果失败，安装 nvidia-container-toolkit
# Ubuntu/Debian:
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. 端口已被占用

```bash
# 查看端口占用
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# 修改 docker-compose.yml 端口映射
ports:
  - "8001:8000"  # 改用 8001
```

#### 3. 卷挂载权限问题

```bash
# Linux 下可能需要修改权限
sudo chown -R $USER:$USER ./core_data

# 或在 Dockerfile 中设置用户
USER 1000:1000
```

#### 4. 容器无法访问宿主机服务

```yaml
# 使用特殊域名
extra_hosts:
  - "host.docker.internal:host-gateway"

# 容器内访问：http://host.docker.internal:3000
```

#### 5. 镜像构建慢

```bash
# 使用构建缓存
docker-compose build

# 使用多阶段构建减小镜像
# 使用 .dockerignore 排除不必要文件
```

### 调试技巧

```bash
# 1. 查看完整日志
docker-compose logs --tail=1000 musetalk

# 2. 进入容器调试
docker-compose exec musetalk bash
# 在容器内：
python3 -c "import torch; print(torch.cuda.is_available())"

# 3. 查看容器资源占用
docker stats $(docker-compose ps -q)

# 4. 检查网络连通性
docker-compose exec musetalk ping database

# 5. 验证卷挂载
docker-compose exec musetalk ls -la /app/models
```

---

## 最佳实践

### 1. 目录结构

```
MuseTalk/
├── docker-compose.yml          # 基础配置
├── docker-compose.dev.yml      # 开发环境
├── docker-compose.prod.yml     # 生产环境
├── Dockerfile                  # 镜像定义
├── .env                        # 环境变量（不提交到 git）
├── .env.example                # 环境变量模板
├── .dockerignore               # 排除文件
└── models/                     # 模型文件（挂载）
```

### 2. .dockerignore

```
# 避免这些文件进入镜像
__pycache__/
*.pyc
.git/
.vscode/
core_data/
models/
*.log
```

### 3. 安全建议

```yaml
# 不要在配置文件中硬编码密钥
environment:
  - API_KEY=${API_KEY}  # 从 .env 读取

# 使用只读卷
volumes:
  - ./models:/app/models:ro

# 限制资源
deploy:
  resources:
    limits:
      memory: 16G
      cpus: '4'
```

### 4. CI/CD 集成

```bash
# GitHub Actions 示例
- name: Build and test
  run: |
    docker-compose build
    docker-compose up -d
    docker-compose exec -T musetalk pytest
    docker-compose down
```

---

## 快速参考

### 一键启动（开发）
```bash
docker-compose -f docker-compose.dev.yml up
```

### 一键启动（生产）
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 完全清理
```bash
docker-compose down -v --rmi all --remove-orphans
```

### 查看实时日志
```bash
docker-compose logs -f --tail=100
```

---

**更多帮助**：
- 官方文档：https://docs.docker.com/compose/
- 配置参考：https://docs.docker.com/compose/compose-file/
