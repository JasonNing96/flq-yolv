
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import warnings
from pathlib import Path

# 忽略字体警告
warnings.filterwarnings('ignore')

# 设置风格
sns.set_theme(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# 尝试加载中文字体 (可选)
try:
    # 假设项目根目录有 SimHei.ttf，如果没有会回退到默认字体
    chinese_font = FontProperties(fname='SimHei.ttf', size=12)
    font_prop = chinese_font
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

def plot_convergence(data_dict, save_path="convergence_curve.png"):
    """
    绘制 mAP50 收敛曲线 (Figure 1)
    """
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("tab10", n_colors=len(data_dict))
    
    # 寻找 Centralized Baseline (如果有)
    central_df = None
    for name, df in data_dict.items():
        if "Centralized" in name:
            central_df = df
            # 假设 Centralized 是按 Epoch 记录的，我们需要将其映射到 Round
            # 简单起见，画一条虚线表示 Best mAP
            best_map = df['mAP50'].max()
            plt.axhline(y=best_map, color='r', linestyle='--', alpha=0.7, label=f"Centralized Best ({best_map:.3f})")
            break

    # 绘制各 FL 实验曲线
    for i, (name, df) in enumerate(data_dict.items()):
        if "Centralized" in name: continue # 已经在上面处理过
        
        # 某些 CSV 可能列名不同，做一下兼容
        x_col = 'round' if 'round' in df.columns else 'epoch'
        y_col = 'mAP50'
        
        if y_col not in df.columns:
            print(f"Skipping {name}: 'mAP50' column not found.")
            continue

        # 平滑曲线 (可选)
        smooth_data = df[y_col].rolling(window=5, min_periods=1).mean()
        
        plt.plot(df[x_col], smooth_data, label=f"{name} (Best: {df[y_col].max():.3f})", 
                 linewidth=2, alpha=0.9, color=colors[i])
        
        # 绘制半透明的原始数据作为背景
        plt.plot(df[x_col], df[y_col], linewidth=0.5, alpha=0.2, color=colors[i])

    plt.title("Convergence Analysis: mAP50 vs. Communication Rounds", fontsize=14)
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("mAP50 Accuracy", fontsize=12)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved convergence plot to {save_path}")

def plot_efficiency(data_dict, save_path="efficiency_tradeoff.png"):
    """
    绘制 效率-精度 权衡图 (Figure 2)
    X轴: 累计通信量 (GB)
    Y轴: Best mAP50
    """
    plt.figure(figsize=(8, 6))
    
    points = []
    
    for name, df in data_dict.items():
        if "Centralized" in name: continue
        
        if 'mAP50' not in df.columns: continue
        
        # 计算总通信量 (Uplink + Downlink)
        # 注意: experiment_data.csv 里记录的是 bits
        # 累计所有轮次
        if 'bits_up_compressed' in df.columns and 'bits_down_compressed' in df.columns:
            total_bits = df['bits_up_compressed'].sum() + df['bits_down_compressed'].sum()
            total_gb = total_bits / 8 / 1024 / 1024 / 1024 # Bits -> GB
        else:
            # 如果没有 bits 列，估算 (假设 standard YOLOv8s ~ 22MB per round * 2 (up/down))
            # 这只是兜底逻辑
            total_gb = len(df) * 22 * 2 / 1024 
            
        best_map = df['mAP50'].max()
        
        points.append({
            'name': name,
            'comm_gb': total_gb,
            'map': best_map,
            'marker': 'o' if 'Ours' in name else 's' # Ours用圆点，Baseline用方块
        })
    
    # 绘制散点
    for p in points:
        plt.scatter(p['comm_gb'], p['map'], s=150, label=p['name'], marker=p['marker'], alpha=0.8, edgecolors='w')
        # 添加文字标签
        plt.text(p['comm_gb'], p['map']+0.005, p['name'], ha='center', va='bottom', fontsize=9)

    plt.title("Efficiency-Accuracy Trade-off", fontsize=14)
    plt.xlabel("Total Communication Cost (GB)", fontsize=12)
    plt.ylabel("Best mAP50 Accuracy", fontsize=12)
    # plt.xscale('log') # 如果差距过大，开启对数坐标
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制 Pareto 前沿 (示意)
    # plt.plot(...) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved efficiency plot to {save_path}")

def main():
    # === 配置你的实验路径 ===
    # 请根据实际情况修改这里的路径
    experiments = {
        "Centralized (Upper Bound)": "results/central_manual_v8s/central_log.csv",
        "Ours (FLQ-Nano)": "results/runs_flq_v6/experiment_data.csv",
        # "FedAvg (Small-Drift)": "results/runs_flq_v6_yolov8s_le1/experiment_data.csv", # 这个是 Drift 严重的
        "Ours (FreezeBN)": "results/runs_flq_v8_yolov8s/experiment_data.csv"  # 正在跑的
    }
    
    # 如果有 v6_nano 的数据（最开始那个0.77的），建议加上
    # experiments["Ours (Nano)"] = "results/runs_flq_v6/experiment_data.csv" # 假设这是你最早跑的那个

    data = load_data(experiments)
    
    if not data:
        print("No data loaded. Check paths.")
        return

    plot_convergence(data, save_path="figure1_convergence.png")
    plot_efficiency(data, save_path="figure2_efficiency.png")

if __name__ == "__main__":
    main()

