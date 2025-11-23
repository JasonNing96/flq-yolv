import os
import random
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib.font_manager import FontProperties
import warnings
import argparse
import torch
from pathlib import Path

# 忽略警告
warnings.filterwarnings('ignore')

# ================== 字体配置 (Font Configuration 仿照 plot.py) ==================
# 优先使用 Noto Sans CJK SC (Bold)
CN_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
if not Path(CN_FONT_PATH).exists():
    # Fallback 1: uming.ttc
    CN_FONT_PATH = '/usr/share/fonts/truetype/arphic/uming.ttc'
    
    # Fallback 2: DroidSansFallbackFull.ttf (常见于某些 Linux 发行版)
    if not Path(CN_FONT_PATH).exists():
        CN_FONT_PATH = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
        
        # Fallback 3: SimHei.ttf (如果用户本地上传了)
        if not Path(CN_FONT_PATH).exists():
             if Path('SimHei.ttf').exists():
                 CN_FONT_PATH = 'SimHei.ttf'

def get_cn_font_props(size=14, weight='normal'):
    """获取中文字体属性对象"""
    if Path(CN_FONT_PATH).exists():
        return FontProperties(fname=CN_FONT_PATH, size=size, weight=weight)
    return None

def visualize_inference(
    model_path='results/central_manual_v8s/best_central.pt',
    image_dir='data/oil_detection_dataset/test/images',
    num_images=9,
    output_path='inference_results.png',
    base_model_path='models/yolov8s.pt' # 用于加载 state_dict 的基础模型
):
    """
    加载模型，对随机选择的测试图片进行推理，并可视化结果。
    """
    
    # 1. 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到: {model_path}")
        return

    # 2. 加载模型
    print(f"正在加载模型: {model_path}...")
    try:
        # 尝试直接加载 (标准 YOLO checkpoint)
        model = YOLO(model_path)
    except Exception:
        print(f"直接加载失败，尝试作为 state_dict 加载到基础模型 {base_model_path}...")
        try:
            # 尝试加载 state_dict
            if not os.path.exists(base_model_path):
                # 尝试自动下载或寻找
                print(f"基础模型 {base_model_path} 不存在，尝试使用 yolov8s.pt")
                base_model_path = 'yolov8s.pt'
            
            model = YOLO(base_model_path)
            state_dict = torch.load(model_path, map_location='cpu')
            
            # 检查是否是 checkpoint 字典但缺少 model 键 (不应该发生如果直接加载失败，但为了保险)
            # 或者它就是纯 state_dict
            if isinstance(state_dict, dict) and 'model' not in state_dict:
                 # 过滤 keys (有时候保存的 keys 可能有前缀，虽然这里似乎是直接保存 model.state_dict())
                model.model.load_state_dict(state_dict)
            elif isinstance(state_dict, dict) and 'model' in state_dict:
                # 这是一个 checkpoint，但之前加载失败了？那可能还是会有问题，但尝试加载内部的 model
                 model.model.load_state_dict(state_dict['model'].float().state_dict())
            
            print("成功加载 state_dict 权重")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return

    # 3. 获取图片列表
    image_files = glob.glob(os.path.join(image_dir, '*.jpg')) + \
                  glob.glob(os.path.join(image_dir, '*.png')) + \
                  glob.glob(os.path.join(image_dir, '*.jpeg'))
    
    if not image_files:
        print(f"错误: 在 {image_dir} 未找到图片")
        return

    # 4. 随机选择图片
    # 确保选择的图片数量不超过实际存在的图片数量
    count = min(num_images, len(image_files))
    selected_images = random.sample(image_files, count)
    print(f"已选择 {count} 张图片进行推理...")

    # 5. 设置中文字体 (使用 plot.py 的逻辑)
    chinese_font = get_cn_font_props(size=12, weight='normal')
    if chinese_font:
        print(f"已加载中文字体: {chinese_font.get_file()}")
    else:
        print(f"警告: 未找到合适的中文字体 ({CN_FONT_PATH})，中文可能无法正确显示。")

    # 6. 推理并绘图
    cols = 3
    rows = (count + cols - 1) // cols
    plt.figure(figsize=(15, 5 * rows))

    for i, img_path in enumerate(selected_images):
        # 推理
        # conf=0.25 是默认置信度阈值，可以根据需要调整
        results = model(img_path, verbose=False, conf=0.25)
        result = results[0]
        
        # 获取绘制了边框的图像 (numpy array, BGR)
        res_plotted = result.plot()
        
        # 转换 BGR 到 RGB 以便 matplotlib 显示
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # 子图
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(res_plotted_rgb)
        
        # 标题
        file_name = os.path.basename(img_path)
        # 截断长文件名
        if len(file_name) > 20:
            file_name = file_name[:17] + "..."
            
        title_text = f"图 {i+1}: {file_name}" # 使用中文前缀
        if chinese_font:
            ax.set_title(title_text, fontproperties=chinese_font)
        else:
            ax.set_title(title_text)
            
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"推理结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO模型推理结果可视化")
    # 默认使用 FLQ 实验中表现最好的模型
    parser.add_argument("--model", type=str, default='results/runs_flq_v8_yolov8s_8bit_3epoch/global_best.pt',
                        help="模型路径 (可以是 checkpoint 或 state_dict)")
    # 默认使用适配过 2 分类的基础模型
    parser.add_argument("--base-model", type=str, default='results/runs_flq_v8_yolov8s/init_adapted.pt', help="基础模型路径 (用于加载 state_dict, 需与权重结构一致)")
    parser.add_argument("--source", type=str, default='data/oil_detection_dataset/test/images', help="图片源目录")
    parser.add_argument("--num", type=int, default=9, help="展示图片数量")
    parser.add_argument("--output", type=str, default='inference_results.png', help="输出图片路径")
    # 移除了 --font 参数，因为现在使用自动检测逻辑
    
    args = parser.parse_args()
    
    visualize_inference(
        model_path=args.model,
        image_dir=args.source,
        num_images=args.num,
        output_path=args.output,
        base_model_path=args.base_model
    )
