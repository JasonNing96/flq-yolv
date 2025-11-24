import warnings
import torch
from ultralytics import YOLO
from pathlib import Path

# 忽略警告
warnings.filterwarnings('ignore')

def train_centralized_upper_bound():
    # ================= 配置区域 =================
    # 1. 模型与数据
    MODEL_NAME = "./models/yolov8s.pt"  # 使用预训练权重作为起点
    DATA_YAML = "/home/njh/project/flq-yolv/data/oil_detection_dataset/data.yaml"
    
    # 2. 输出路径
    PROJECT_DIR = "results/central_baseline_upperbound"
    RUN_NAME = "yolov8s_full_200e"
    
    # 3. 训练超参数 (旨在探索性能上限)
    HYPER_PARAMS = {
        "epochs": 200,          # 充分训练
        "patience": 50,         # 早停轮数
        "batch": 32,            # 批次大小 (显存不足可改为 16)
        "imgsz": 640,           # 输入分辨率
        "device": "0",          # 使用第一块 GPU
        "workers": 8,           # 数据加载线程
        "pretrained": True,     # 使用 COCO 预训练权重
        "optimizer": "auto",    # 自动选择优化器 (通常是 SGD)
        "lr0": 0.01,            # 初始学习率
        "lrf": 0.01,            # 最终学习率 (lr0 * lrf)
        "momentum": 0.937,      # 动量
        "weight_decay": 0.0005, # 权重衰减
        "warmup_epochs": 3.0,   # 预热轮数
        "box": 7.5,             # 边框损失增益
        "cls": 0.5,             # 分类损失增益
        "dfl": 1.5,             # DFL 损失增益
        "plots": True,          # 自动绘制曲线
        "val": True,            # 训练期间验证
        "save": True,           # 保存 Checkpoint
        "exist_ok": True,       # 允许覆盖
    }
    # ===========================================

    print(f"🚀 开始集中式训练上限探索: {MODEL_NAME}")
    print(f"📁 数据集配置: {DATA_YAML}")
    print(f"💾 结果保存至: {PROJECT_DIR}/{RUN_NAME}")

    # 确保输出目录存在
    Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = YOLO(MODEL_NAME)

    # 开始训练
    try:
        results = model.train(
            data=DATA_YAML,
            project=PROJECT_DIR,
            name=RUN_NAME,
            **HYPER_PARAMS
        )
        
        print(f"✅ 训练完成!")
        print(f"📊 最佳 mAP@50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"💾 最佳模型路径: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    # 检查 GPU
    if not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到 GPU，训练将非常缓慢！")
    else:
        print(f"🔥 检测到 GPU: {torch.cuda.get_device_name(0)}")
        
    train_centralized_upper_bound()
