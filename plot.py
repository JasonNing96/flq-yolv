
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import warnings
from pathlib import Path
import numpy as np
import glob
import random
import cv2
from ultralytics import YOLO

# 忽略字体警告
warnings.filterwarnings('ignore')

# 设置风格
sns.set_theme(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# 尝试加载中文字体 (可选，用于显示中文标签)
try:
    # 尝试常见的中文字体路径，或项目目录下的 SimHei.ttf
    font_path = 'SimHei.ttf' 
    if not Path(font_path).exists():
         # 如果没有本地字体，尝试系统字体 (Ubuntu)
         font_path = '/usr/share/fonts/truetype/arphic/uming.ttc'
    
    if Path(font_path).exists():
        chinese_font = FontProperties(fname=font_path, size=12)
        font_prop = chinese_font
        print(f"Loaded font: {font_path}")
    else:
        font_prop = None
        print("No chinese font found, using default.")
except:
    font_prop = None

def load_data(paths: dict):
    """
    加载不同实验的 CSV 数据
    paths: dict, key=Experiment Name, value=Path to CSV
    """
    data = {}
    for name, path in paths.items():
        p = Path(path)
        if not p.exists():
            print(f"Warning: File not found: {path}")
            continue
        try:
            df = pd.read_csv(p)
            data[name] = df
            print(f"Loaded {name}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    return data

def plot_convergence(data_dict, save_path="figure1_convergence.png"):
    """
    绘制 mAP50 收敛曲线 (Figure 1)
    """
    plt.figure(figsize=(10, 6))
    
    # 自定义颜色映射，确保 Ours 是显眼的颜色(如红色或蓝色)
    colors = sns.color_palette("tab10", n_colors=len(data_dict))
    color_map = {}
    
    # 寻找 Centralized Baseline (Upper Bound)
    central_best = 0
    for name, df in data_dict.items():
        if "Centralized" in name:
            # Centralized 通常按 Epoch 记录，我们需要获取其最佳值作为参考线
            central_best = df['mAP50'].max()
            plt.axhline(y=central_best, color='r', linestyle='--', alpha=0.7, linewidth=1.5, 
                       label=f"Centralized Bound ({central_best:.3f})")
            break
            
    # 绘制各 FL 实验曲线
    idx = 0
    for name, df in data_dict.items():
        if "Centralized" in name: continue 
        
        # 确定X轴和Y轴
        x_col = 'round' if 'round' in df.columns else 'epoch'
        y_col = 'mAP50'
        
        if y_col not in df.columns:
            print(f"Skipping {name}: '{y_col}' column not found.")
            continue

        # 颜色分配
        color = colors[idx]
        idx += 1
        
        # 线宽和样式：Ours 加粗
        linewidth = 2.5 if "Ours" in name else 1.5
        alpha = 1.0 if "Ours" in name else 0.7
        
        # 平滑曲线 (Rolling Mean) 使趋势更清晰
        smooth_data = df[y_col].rolling(window=5, min_periods=1).mean()
        
        plt.plot(df[x_col], smooth_data, label=f"{name} (Max: {df[y_col].max():.3f})", 
                 linewidth=linewidth, alpha=alpha, color=color)
        
        # 绘制半透明的原始数据作为背景（阴影效果）
        plt.plot(df[x_col], df[y_col], linewidth=0.5, alpha=0.2, color=color)

    plt.title("Figure 1: Convergence Analysis (mAP50 vs. Rounds)", fontsize=14, fontweight='bold')
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("mAP50 Accuracy", fontsize=12)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 200) # 根据实际Round调整
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved convergence plot to {save_path}")

def plot_efficiency(data_dict, save_path="figure2_efficiency.png"):
    """
    绘制 效率-精度 权衡图 (Figure 2) - 帕累托图
    X轴: 累计通信量 (GB) (Log Scale)
    Y轴: Best mAP50
    """
    plt.figure(figsize=(9, 7))
    
    points = []
    
    for name, df in data_dict.items():
        if "Centralized" in name: continue
        
        if 'mAP50' not in df.columns: continue
        
        # 1. 计算总通信量 (Total Communication Cost)
        # 优先使用 experiment_data.csv 中的 bits 数据
        if 'bits_up_compressed' in df.columns and 'bits_down_compressed' in df.columns:
            # Sum of all rounds (Up + Down)
            total_bits = df['bits_up_compressed'].sum() + df['bits_down_compressed'].sum()
            total_gb = total_bits / 8 / 1024 / 1024 / 1024 # Bits -> Bytes -> KB -> MB -> GB
        else:
            # 兜底估算：如果没有 bits 数据，假设它是标准 YOLOv8s
            # 这里的估算系数需要根据你的模型大小调整
            # 假设: Standard Model ~ 22.5MB 参数量
            model_size_mb = 22.5 if "YOLOv8s" in name else 6.0 # Nano ~ 6MB
            rounds = len(df)
            # Up + Down per round
            total_gb = (model_size_mb * 2 * rounds) / 1024 
            
        best_map = df['mAP50'].max()
        
        points.append({
            'name': name,
            'comm_gb': total_gb,
            'map': best_map,
            'marker': 'o' if 'Ours' in name else 's', # Ours用圆点，Baseline用方块
            'color': 'red' if 'Ours' in name else 'gray',
            'size': 200 if 'Ours' in name else 100
        })
    
    # 绘制散点
    for p in points:
        plt.scatter(p['comm_gb'], p['map'], s=p['size'], c=p['color'], 
                    marker=p['marker'], alpha=0.8, edgecolors='k', label=p['name'])
        
        # 添加文字标签 (稍微偏移一点以免遮挡)
        offset_y = 0.005
        plt.text(p['comm_gb'], p['map'] + offset_y, p['name'], 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title("Figure 2: Efficiency-Accuracy Trade-off", fontsize=14, fontweight='bold')
    plt.xlabel("Total Communication Cost (GB) [Log Scale]", fontsize=12)
    plt.ylabel("Best mAP50 Accuracy", fontsize=12)
    
    # 使用对数坐标，因为 Nano 量化后通信量可能比 Standard 低几个数量级
    plt.xscale('log') 
    
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    
    # 移除重复的 legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved efficiency plot to {save_path}")

def generate_figure3(model_path, val_images_dir, output_dir="figure3_viz", num_samples=4):
    """
    生成 Figure 3: 目标检测可视化结果
    1. 加载训练好的模型
    2. 从验证集中随机抽取图片
    3. 进行推理并画框
    4. 拼接保存
    """
    print(f"\nGenerating Figure 3 using model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return

    # 确保输出目录存在
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    # 加载模型
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # 获取图片列表
    # 支持 jpg, png, jpeg
    img_patterns = [f"{val_images_dir}/*.jpg", f"{val_images_dir}/*.png", f"{val_images_dir}/*.jpeg"]
    img_files = []
    for pattern in img_patterns:
        img_files.extend(glob.glob(pattern))
    
    if not img_files:
        print(f"Error: No images found in {val_images_dir}")
        return
        
    # 随机采样
    if len(img_files) > num_samples:
        selected_imgs = random.sample(img_files, num_samples)
    else:
        selected_imgs = img_files
        
    # 运行推理
    results = model(selected_imgs, verbose=False)
    
    # 保存结果
    plot_paths = []
    for i, r in enumerate(results):
        # r.plot() 返回的是 BGR numpy 数组
        im_array = r.plot()
        
        # 保存单张图
        save_file = out_path / f"pred_{i}.jpg"
        cv2.imwrite(str(save_file), im_array)
        plot_paths.append(str(save_file))
        
    print(f"Saved {len(plot_paths)} visualization images to {output_dir}")
    
    # 尝试拼接图片 (2x2 grid)
    if len(plot_paths) == 4:
        try:
            img1 = cv2.imread(plot_paths[0])
            img2 = cv2.imread(plot_paths[1])
            img3 = cv2.imread(plot_paths[2])
            img4 = cv2.imread(plot_paths[3])
            
            # 调整大小一致
            h, w = img1.shape[:2]
            img2 = cv2.resize(img2, (w, h))
            img3 = cv2.resize(img3, (w, h))
            img4 = cv2.resize(img4, (w, h))
            
            top = np.hstack((img1, img2))
            bottom = np.hstack((img3, img4))
            grid = np.vstack((top, bottom))
            
            cv2.imwrite("figure3_detection_results.png", grid)
            print("Saved combined Figure 3 to figure3_detection_results.png")
        except Exception as e:
            print(f"Could not create grid image: {e}")

def main():
    # ================= 配置区域 =================
    
    # 1. 实验数据路径 (CSV)
    experiments = {
        "Centralized (Upper Bound)": "results/central_manual_v8s/central_log.csv",
        
        "FedAvg (Baseline)": "results/runs_flq_v8_yolov8s/experiment_data.csv",
        
        "Ours (FLQ-Nano)": "results/runs_flq_v8_yolov8n_8bit_1epoch/experiment_data.csv",
        
        # 对比组：Small模型 + 8bit量化
        "Comparison (Small-8bit)": "results/runs_flq_v6_yolov8s_8bit_1epochs/experiment_data.csv"
    }
    
    # 2. 模型路径 (用于 Figure 3)
    # 使用你效果最好的 FLQ-Nano 模型
    best_model_path = "results/runs_flq_v8_yolov8n_8bit_1epoch/global_best.pt"
    
    # 3. 验证集图片路径 (从你的 data.yaml 得知)
    val_images_path = "data/oil_detection_dataset/valid/images" 
    
    # ================= 执行区域 =================
    
    # 1. 加载数据
    print("--- Loading Data ---")
    data = load_data(experiments)
    
    if data:
        # 2. 绘制 Figure 1 (收敛)
        print("\n--- Plotting Figure 1 ---")
        plot_convergence(data, save_path="figure1_convergence.png")
        
        # 3. 绘制 Figure 2 (效率)
        print("\n--- Plotting Figure 2 ---")
        plot_efficiency(data, save_path="figure2_efficiency.png")
    
    # 4. 生成 Figure 3 (可视化)
    print("\n--- Generating Figure 3 ---")
    generate_figure3(best_model_path, val_images_path, num_samples=4)
    print("\nDone! All figures generated.")

if __name__ == "__main__":
    main()

