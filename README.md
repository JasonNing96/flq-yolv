# FLQ-YOLOv: 联邦量化学习框架

基于 YOLOv8 的联邦学习框架，支持梯度量化和压缩，适用于边缘设备（如 Jetson）的分布式训练。

## 核心特性

- **FreezeBN**: 冻结 BatchNorm 统计量，解决 Non-IID 数据导致的 BN 偏移
- **FLQ 压缩**: 支持 1/4/8-bit 梯度量化，降低通信开销
- **Error Feedback**: 误差反馈机制，提升量化精度
- **分布式架构**: Client-Server 架构，支持 Docker 部署

## 快速开始

### 安装依赖

```bash
pip install -r app/requirements.txt
```

### 单机训练

```bash
python flq_yolov_v8.py \
    --clients data/client1/oil.yaml data/client2/oil.yaml data/client3/oil.yaml \
    --val-data data/client1/oil.yaml \
    --model models/yolov8n.pt \
    --rounds 200 \
    --local-epochs 2 \
    --bits 8 \
    --batch 4 \
    --device cuda:0
```

### 分布式训练

#### 1. 配置

编辑 `app/flq_config.yaml`:

```yaml
training:
  rounds: 200
  clients_per_round: 3
  local_epochs: 2

quantization:
  enabled: true
  bits: 8
  error_feedback_enabled: true

model:
  name: models/yolov8n.pt
  device: cuda:0

server:
  host: 0.0.0.0
  port: 8087
  aggregation_mode: flq-fed  # flq-fed 或 fedavg

client:
  batch_size: 4
  workers: 0
```

#### 2. 启动

**方式一：一键启动（单机多进程）**
```bash
python -m app.runner train
```

**方式二：手动启动**

```bash
# 终端1: 启动服务器
python -m app.runner server

# 终端2/3/4: 启动客户端
python -m app.runner client --id 1
python -m app.runner client --id 2
python -m app.runner client --id 3
```

## Docker 部署

### 标准环境 (x86/GPU)

```bash
cd app
docker build -f Dockerfile -t flq-fed:latest ..
docker-compose up -d
```

### Jetson 设备

```bash
cd app
# 修改 Dockerfile.jetson 中的 L4T tag 匹配您的 JetPack 版本
docker build -f Dockerfile.jetson -t flq-fed-jetson:latest ..
docker-compose -f docker-compose.jetson.yml up -d
```

详细 Jetson 部署说明见 `app/JETSON_DEPLOY.md`

## 输出

- **单机版**: `results/runs_flq_v8_*/experiment_data.csv`
- **分布式**: `outputs/server/checkpoints/` 和 `outputs/client*/runs/`

## 项目结构

```
flq-yolv/
├── app/                    # 分布式实现
│   ├── client.py          # 客户端
│   ├── server.py          # 服务器
│   ├── flq_config.yaml    # 配置文件
│   └── ...
├── flq_yolov_v8.py        # 单机版核心代码
└── data/                  # 数据集
```

## 常见问题

- **CUDA OOM**: 减小 `batch_size` 或使用 `yolov8n.pt`
- **Jetson 构建失败**: 检查 JetPack 版本，修改 `Dockerfile.jetson` 中的 L4T tag
- **连接失败**: 检查防火墙和端口设置
