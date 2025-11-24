# 基础镜像：使用带 CUDA 支持的 PyTorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
# 注意：我们在构建上下文的根目录运行 docker build，所以复制整个项目
COPY . /workspace/

# 暴露端口 (Server用)
EXPOSE 8087

# 默认命令 (可以被 docker-compose 覆盖)
CMD ["python", "-m", "app.runner", "train"]

