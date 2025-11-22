#!/bin/bash

# 这是一个记录如何复现图 2.2 数据的脚本参考
# 包含三种算法的运行命令

# 1. FLQ (本算法)
# 对应文件: results/results_mnist_bbit.xlsx
echo "Running FLQ (bbit)..."
python3 flq_fed_v4.py --dataset mnist --mode bbit --iters 800 --M 10 --b 8

# 2. LAQ (8-bit 随机量化)
# 对应文件: results/results_mnist_laq8.xlsx
echo "Running LAQ..."
python3 flq_fed_v4.py --dataset mnist --mode laq8 --iters 800 --M 10

# 3. QGD (在此处作为全精度 FedAvg 基线，或未压缩基线)
# 对应文件: results/results_mnist_qgd.xlsx (通常对应 fedavg)
echo "Running QGD (FedAvg)..."
python3 flq_fed_v4.py --dataset mnist --mode fedavg --iters 800 --M 10

