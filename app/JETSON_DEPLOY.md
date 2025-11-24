# Jetson 部署指南

本指南专门针对 NVIDIA Jetson 设备的部署。

## 前置条件

### 1. 系统要求

- **JetPack 5.1+** (推荐) 或 JetPack 4.6+
- **Docker** 和 **nvidia-docker2** 已安装
- **至少 8GB 内存** (推荐 16GB+)
- **足够的存储空间** (至少 20GB 可用)

### 2. 检查系统版本

```bash
# 检查 JetPack 版本
cat /etc/nv_tegra_release

# 检查 CUDA 版本
nvcc --version

# 检查 Docker
docker --version
docker run --rm --runtime nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 3. 安装 nvidia-docker2 (如果未安装)

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 快速部署

### 方法 1: Docker Compose (推荐)

```bash
# 1. 进入项目目录
cd /path/to/flq-yolv/app

# 2. 修改 docker-compose.jetson.yml 中的基础镜像 tag
# 根据您的 JetPack 版本选择：
# - JetPack 5.1: r35.2.1-pth2.0-py3
# - JetPack 4.6: r32.7.1-pth1.10-py3

# 3. 构建镜像
docker build -f Dockerfile.jetson -t flq-fed-jetson:latest ..

# 4. 启动服务
docker-compose -f docker-compose.jetson.yml up -d

# 5. 查看日志
docker-compose -f docker-compose.jetson.yml logs -f
```

### 方法 2: 手动 Docker 运行

#### 启动 Server

```bash
docker run -d --name flq-server \
  --runtime nvidia \
  -p 8087:8087 \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -e NVIDIA_VISIBLE_DEVICES=all \
  flq-fed-jetson:latest \
  python3 -m app.runner server
```

#### 启动 Client

```bash
docker run -d --name flq-client-1 \
  --runtime nvidia \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  flq-fed-jetson:latest \
  python3 -m app.runner client --id 1 --server http://<SERVER_IP>:8087
```

## 多设备部署

### 场景：1 个 Server + N 个 Client (不同 Jetson)

#### 在 Jetson A (Server) 上：

```bash
# 1. 启动 Server
docker run -d --name flq-server \
  --runtime nvidia \
  -p 8087:8087 \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  flq-fed-jetson:latest \
  python3 -m app.runner server

# 2. 获取 IP 地址
hostname -I
```

#### 在 Jetson B/C/D (Client) 上：

```bash
# 替换 <JETSON_A_IP> 为 Server 的 IP 地址
docker run -d --name flq-client-1 \
  --runtime nvidia \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  flq-fed-jetson:latest \
  python3 -m app.runner client --id 1 --server http://<JETSON_A_IP>:8087
```

## 性能优化

### 1. 内存优化

如果遇到 OOM 错误：

```yaml
# app/flq_config.yaml
client:
  batch_size: 2  # 减小批次大小
  workers: 0     # 禁用多进程数据加载

model:
  device: cpu    # 如果 GPU 显存不足，使用 CPU
```

### 2. 通信优化

```yaml
quantization:
  enabled: true
  bits: 4        # 使用 4-bit 量化减少通信量
```

### 3. 模型选择

使用较小的模型：
- `yolov8n.pt` (推荐)
- 避免使用 `yolov8m.pt` 或更大的模型

## 监控和调试

### 查看容器状态

```bash
docker ps
docker stats
```

### 查看日志

```bash
# Server 日志
docker logs -f flq-server

# Client 日志
docker logs -f flq-client-1
```

### 进入容器调试

```bash
docker exec -it flq-server bash
docker exec -it flq-client-1 bash
```

### 监控 GPU

```bash
# 在宿主机上
watch -n 1 nvidia-smi

# 在容器内
docker exec flq-server nvidia-smi
```

## 常见问题

### 1. 构建失败：找不到基础镜像

**问题**：`docker build` 时提示找不到 `nvcr.io/nvidia/l4t-pytorch:xxx`

**解决**：
- 检查 JetPack 版本：`cat /etc/nv_tegra_release`
- 访问 [NVIDIA NGC](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=scoreDESC&query=l4t-pytorch) 查看可用的 tag
- 修改 `Dockerfile.jetson` 中的 `FROM` 行

### 2. 运行时错误：CUDA not available

**问题**：容器内无法访问 GPU

**解决**：
```bash
# 检查 nvidia-docker2
docker run --rm --runtime nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 确保使用 --runtime nvidia
docker run --runtime nvidia ...
```

### 3. 内存不足

**问题**：`CUDA out of memory`

**解决**：
- 减小 `batch_size` (2 或 1)
- 使用 CPU 模式
- 关闭其他占用 GPU 的程序

### 4. 网络连接失败

**问题**：Client 无法连接到 Server

**解决**：
- 检查防火墙：`sudo ufw status`
- 确认端口开放：`sudo ufw allow 8087`
- 检查 IP 地址是否正确
- 在 Server 上测试：`curl http://localhost:8087/status`

## 性能基准

在 Jetson AGX Xavier (JetPack 5.1) 上的参考性能：

- **模型**: YOLOv8n
- **Batch Size**: 4
- **量化**: 8-bit
- **训练速度**: ~2-3 秒/epoch (640x640)
- **显存占用**: ~2-3 GB

## 参考资源

- [NVIDIA Jetson Docker](https://github.com/dusty-nv/jetson-containers)
- [L4T PyTorch Images](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=scoreDESC&query=l4t-pytorch)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)

