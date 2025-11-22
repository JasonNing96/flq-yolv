#!/bin/bash

# 1. 确保在 flq_simulation 目录下
cd "$(dirname "$0")"

# 2. 运行绘图脚本
# --excel_dir results: 指定数据文件目录
# --dataset mnist: 指定数据集
# --save: 保存图片而不是显示
# --modes qgd laq8 bbit: 指定要绘制的曲线（对应 results/results_mnist_{mode}.xlsx）

echo "正在绘制图表..."
python3 plot_flq_fed.py \
    --excel_dir results \
    --dataset mnist \
    --modes qgd laq8 bbit \
    --max_iter 800 \
    --save

echo "完成！图片已保存到 figures/Fig2_mnist.png"

