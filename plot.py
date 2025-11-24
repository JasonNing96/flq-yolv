import argparse
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

# 设置全局风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False

# ================== 字体配置 (Font Configuration) ==================
# 优先使用 Noto Sans CJK SC (Bold)
CN_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
if not Path(CN_FONT_PATH).exists():
    # Fallback
    CN_FONT_PATH = '/usr/share/fonts/truetype/arphic/uming.ttc'

def get_cn_font_props(size=14, weight='bold'):
    """获取中文字体属性对象"""
    if Path(CN_FONT_PATH).exists():
        return FontProperties(fname=CN_FONT_PATH, size=size, weight=weight)
    return None

# ================== 通用工具函数 ==================
def load_data(paths: dict):
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

def ensure_dir(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

# ================== 中文绘图函数 (CN Functions) ==================
# 完全独立的中文绘图逻辑，确保风格统一

def get_method_style(name):
    """根据方法名称返回颜色、线条样式和标签"""
    # 判断方法类型
    if "集中式" in name or "Centralized" in name:
        return {
            'color': '#808080',  # 灰色
            'linestyle': '--',   # 虚线
            'linewidth': 2.5,
            'label': '集中式基准'
        }
    elif "本文" in name or "Ours" in name:
        return {
            'color': '#2ca02c',  # 绿色
            'linestyle': '-',    # 实线
            'linewidth': 3.5,    # 加粗
            'label': 'SA-FLQ'
        }
    elif "量化" in name or "Quant" in name:
        return {
            'color': '#ff7f0e',  # 橙色
            'linestyle': '--',   # 虚线
            'linewidth': 2.5,
            'label': 'FedAvg-量化'
        }
    else:  # FedAvg
        return {
            'color': '#1f77b4',  # 蓝色
            'linestyle': '-',    # 实线
            'linewidth': 2.5,
            'label': 'FedAvg'
        }

def plot_loss_cn(data_dict, save_path="results/figure/loss_convergence_cn.png"):
    ensure_dir(save_path)
    
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(12, 8))
    
    # 字体配置
    label_font = get_cn_font_props(size=20, weight='bold')
    legend_font = get_cn_font_props(size=14, weight='bold')
    tick_font = get_cn_font_props(size=16, weight='bold')
    
    for name, df in data_dict.items():
        if "Centralized" in name or "集中式" in name: continue 

        x_col = 'round' if 'round' in df.columns else 'epoch'
        y_col = 'avg_loss' if 'avg_loss' in df.columns else ('loss' if 'loss' in df.columns else None)
        if not y_col: continue

        # 获取样式
        style = get_method_style(name)
        is_ours = "本文" in name or "Ours" in name
        zorder = 10 if is_ours else 5
        
        # 平滑
        smooth = df[y_col].rolling(window=10, min_periods=1).mean()
        
        plt.plot(df[x_col], smooth, label=style['label'], 
                color=style['color'], linestyle=style['linestyle'],
                linewidth=3.0, alpha=1.0, zorder=zorder)
        # 背景淡线
        plt.plot(df[x_col], df[y_col], color=style['color'], 
                linewidth=0.8, alpha=0.1, zorder=zorder-1)

    # 设置标签和标题
    # plt.title("训练损失收敛曲线 (溢油检测数据集)", fontproperties=title_font, pad=15) # 取消标题
    plt.xlabel("通信轮次", fontproperties=label_font)
    plt.ylabel("训练损失 (平滑处理)", fontproperties=label_font)
    
    # 美化刻度与边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2.0)
    
    ax.tick_params(which='major', width=2.0, length=6, labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(tick_font)
    
    # 图例
    plt.legend(loc='upper right', prop=legend_font, 
               frameon=True, fancybox=True, shadow=True, framealpha=1.0, 
               facecolor='white', edgecolor='#cccccc', borderpad=0.8)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
    plt.xlim(0, 200)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved CN Loss Plot to {save_path}")

def plot_convergence_cn(data_dict, save_path="results/figure/map_convergence_cn.png"):
    ensure_dir(save_path)
    
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(12, 8))
    
    label_font = get_cn_font_props(size=20, weight='bold')
    legend_font = get_cn_font_props(size=14, weight='bold')
    tick_font = get_cn_font_props(size=16, weight='bold')

    # 1. 基准线 (统一使用 0.775)
    plt.axhline(y=0.7661, color='#ff6666', linestyle='--', linewidth=2.5, alpha=0.9,
               label='集中式基准 (上限)')

    # 2. 实验曲线
    for name, df in data_dict.items():
        if "集中式" in name or "Centralized" in name: continue
        
        x_col = 'round' if 'round' in df.columns else 'epoch'
        y_col = 'mAP50'
        if y_col not in df.columns: continue
        
        # 获取样式
        style = get_method_style(name)
        is_ours = "本文" in name or "Ours" in name
        zorder = 10 if is_ours else 5
        
        smooth = df[y_col].rolling(window=8, min_periods=1).mean()
        
        plt.plot(df[x_col], smooth, label=style['label'], 
                color=style['color'], linestyle=style['linestyle'],
                linewidth=3.0, alpha=1.0, zorder=zorder)
        
        plt.plot(df[x_col], df[y_col], color=style['color'], 
                linewidth=1, alpha=0.15, zorder=zorder-1)

    plt.xlabel("通信轮次", fontproperties=label_font)
    plt.ylabel("mAP@0.50 精度", fontproperties=label_font)
    
    # 美化刻度与边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2.0)
    
    ax.tick_params(which='major', width=2.0, length=6, labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(tick_font)
    
    plt.legend(loc='lower right', prop=legend_font, 
               frameon=True, fancybox=True, shadow=True, framealpha=1.0, 
               facecolor='white', edgecolor='#cccccc', borderpad=0.8)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
    plt.xlim(0, 200)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved CN Convergence Plot to {save_path}")

def plot_efficiency_cn(data_dict, save_path="results/figure/efficiency_tradeoff_cn.png"):
    ensure_dir(save_path)
    
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(10, 8))
    
    label_font = get_cn_font_props(size=20, weight='bold')
    text_font = get_cn_font_props(size=13, weight='bold') 
    legend_font = get_cn_font_props(size=14, weight='bold')
    tick_font = get_cn_font_props(size=16, weight='bold')
    
    points = []
    for name, df in data_dict.items():
        if "集中式" in name or "Centralized" in name: continue
        if 'mAP50' not in df.columns: continue
        
        # 计算通信量
        if 'bits_up_compressed' in df.columns:
            total_bits = df['bits_up_compressed'].sum() + df['bits_down_compressed'].sum()
            total_gb = total_bits / 8 / (1024**3)
        else:
            model_size_mb = 6.0 if "YOLOv8n" in name else 22.5
            if "8位" in name or "8-bit" in name: model_size_mb /= 4
            rounds = len(df)
            total_gb = (model_size_mb * 2 * rounds) / 1024
            
        best_map = df['mAP50'].max()
        style = get_method_style(name)
        is_ours = "本文" in name or "Ours" in name
        
        points.append({
            'name': name,
            'comm_gb': total_gb,
            'map': best_map,
            'marker': 'o' if is_ours else 'D',
            'color': style['color'],
            'size': 350 if is_ours else 200,
            'label': style['label']
        })
        
    for p in points:
        plt.scatter(p['comm_gb'], p['map'], s=p['size'], c=p['color'], 
                    marker=p['marker'], alpha=0.95, edgecolors='white', linewidth=2,
                    label=p['label'], zorder=10)
        
        # 标注文字
        t = plt.text(p['comm_gb'], p['map'] + 0.006, p['label'], 
                 ha='center', va='bottom', fontproperties=text_font, zorder=15)

    # plt.title("通信效率与模型精度权衡分析", fontproperties=title_font, pad=15)
    plt.xlabel("总通信开销 (GB) [对数坐标]", fontproperties=label_font)
    plt.ylabel("最佳 mAP50 精度", fontproperties=label_font)
    
    plt.xscale('log')
    
    # 美化刻度与边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2.0)
    
    ax.tick_params(which='major', width=2.0, length=6, labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(tick_font)
    
    # 效率图保留全网格更好看
    plt.grid(True, which="both", linestyle='--', alpha=0.4, color='gray')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right', 
               prop=legend_font, frameon=True, fancybox=True, shadow=True, 
               framealpha=1.0, facecolor='white', edgecolor='#cccccc', borderpad=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved CN Efficiency Plot to {save_path}")

def plot_convergence_loss_cn(data_dict, save_path="results/figure/convergence_loss_cn.png"):
    ensure_dir(save_path)
    
    plt.style.use('seaborn-v0_8-white')
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 字体配置
    label_font = get_cn_font_props(size=20, weight='bold')
    legend_font = get_cn_font_props(size=14, weight='bold')
    tick_font = get_cn_font_props(size=16, weight='bold')
    
    ax2 = ax1.twinx()

    lines = []
    labels = []

    # 1. 绘制 Upper Bound (集中式基准上限)
    # 淡红色虚线
    l_bound = ax1.axhline(y=0.7661, color='#ff6666', linestyle='--', linewidth=2.5, alpha=0.9)
    lines.append(l_bound)
    labels.append('集中式基准 (上限)')

    # 2. 绘制各方法曲线
    for name, df in data_dict.items():
        if "集中式" in name or "Centralized" in name: continue
        
        x_col = 'round' if 'round' in df.columns else 'epoch'
        
        y_map = df['mAP50'].rolling(window=5, min_periods=1).mean() if 'mAP50' in df.columns else None
        
        loss_col = 'avg_loss' if 'avg_loss' in df.columns else ('loss' if 'loss' in df.columns else None)
        y_loss = df[loss_col].rolling(window=5, min_periods=1).mean() if loss_col else None

        if y_map is None: continue

        style = get_method_style(name)
        color = style['color']
        label_base = style['label']

        # 绘制 mAP (实线, 左轴)
        l1, = ax1.plot(df[x_col], y_map, color=color, linestyle='-', linewidth=3.0)
        lines.append(l1)
        labels.append(label_base) # 图例只显示方法名，不区分 mAP/Loss

        # 绘制 Loss (点划线, 右轴) - 不加入图例
        if y_loss is not None:
            ax2.plot(df[x_col], y_loss, color=color, linestyle='--', linewidth=2.5, alpha=0.6)

    # 设置轴标签 (取消标题)
    ax1.set_xlabel("通信轮次", fontproperties=label_font)
    ax1.set_ylabel("mAP@0.50 精度 (实线)", fontproperties=label_font)
    ax2.set_ylabel("训练损失 (虚线)", fontproperties=label_font, rotation=270, labelpad=25)
    
    # 美化刻度与边框
    for ax in [ax1, ax2]:
        # 去掉顶部边框
        ax.spines['top'].set_visible(False)
        # 加粗其他边框
        for spine in ['bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(2.0)
        
        ax.tick_params(which='major', width=2.0, length=6, labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(tick_font)

    # 网格线优化：只显示左轴的水平网格
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
    ax1.grid(False, axis='x')
    ax2.grid(False)

    # 图例优化：放在图表内部，美观样板
    ax1.legend(lines, labels, loc='center right', prop=legend_font, 
               frameon=True, fancybox=True, shadow=True, framealpha=1.0, 
               facecolor='white', edgecolor='#cccccc', borderpad=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved CN Cross Plot to {save_path}")

def plot_convergence_loss_en(data_dict, save_path="results/figure/convergence_loss_en.png"):
    ensure_dir(save_path)
    
    plt.style.use('seaborn-v0_8-white')
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    ax2 = ax1.twinx()

    lines = []
    labels = []

    # 1. Upper Bound
    l_bound = ax1.axhline(y=0.7661, color='#ff6666', linestyle='--', linewidth=2.5, alpha=0.9)
    lines.append(l_bound)
    labels.append('Centralized (Upper Bound)')

    # 2. Experiments
    # Note: For EN, keys are English. get_method_style handles "Centralized", "Ours", "Quant" checks.
    # But it returns Chinese labels. We will manually handle labels here or use a simplified approach.
    
    for name, df in data_dict.items():
        if "Centralized" in name: continue
        
        x_col = 'round' if 'round' in df.columns else 'epoch'
        y_map = df['mAP50'].rolling(window=5, min_periods=1).mean() if 'mAP50' in df.columns else None
        
        loss_col = 'avg_loss' if 'avg_loss' in df.columns else ('loss' if 'loss' in df.columns else None)
        y_loss = df[loss_col].rolling(window=5, min_periods=1).mean() if loss_col else None

        if y_map is None: continue

        style = get_method_style(name)
        color = style['color']
        # Simple English Label: first part before '('
        # e.g. "FedAvg (YOLOv8s, 32-bit)" -> "FedAvg"
        label_en = name.split('(')[0].strip()
        # Handle "Ours" if specific replacement needed, otherwise use name part
        if "Ours" in name: label_en = "SA-FLQ" # Or "Ours"

        # Plot mAP (Solid, Left Axis)
        l1, = ax1.plot(df[x_col], y_map, color=color, linestyle='-', linewidth=3.0)
        lines.append(l1)
        labels.append(label_en)

        # Plot Loss (Dashed, Right Axis)
        if y_loss is not None:
            ax2.plot(df[x_col], y_loss, color=color, linestyle='--', linewidth=2.5, alpha=0.6)

    # Labels
    ax1.set_xlabel("Communication Rounds", fontsize=20, fontweight='bold')
    ax1.set_ylabel("mAP@0.50 (Solid)", fontsize=20, fontweight='bold')
    ax2.set_ylabel("Training Loss (Dashed)", fontsize=20, fontweight='bold', rotation=270, labelpad=25)
    
    # Spines & Ticks
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        for spine in ['bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(2.0)
        
        ax.tick_params(which='major', width=2.0, length=6, labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    # Grid (Left Y only)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
    ax1.grid(False, axis='x')
    ax2.grid(False)

    # Legend
    ax1.legend(lines, labels, loc='center right', fontsize=14, 
               frameon=True, fancybox=True, shadow=True, framealpha=1.0, 
               facecolor='white', edgecolor='#cccccc', borderpad=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved EN Cross Plot to {save_path}")

# ================== 英文绘图函数 (EN Functions - Keep Original Logic) ==================
# 保持之前的英文逻辑或稍作简化

def plot_loss_en(data_dict, save_path="results/figure/loss_convergence.png"):
    ensure_dir(save_path)
    
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(12, 8))
    
    colors = sns.color_palette("tab10", n_colors=len(data_dict))
    idx = 0
    
    for name, df in data_dict.items():
        if "Centralized" in name: continue
        x_col = 'round' if 'round' in df.columns else 'epoch'
        y_col = 'avg_loss' if 'avg_loss' in df.columns else 'loss'
        if y_col not in df.columns: continue
        
        color = colors[idx]; idx += 1
        
        # Smooth
        plt.plot(df[x_col], df[y_col].rolling(5).mean(), label=name, color=color, linewidth=3.0)
        # Background
        plt.plot(df[x_col], df[y_col], color=color, alpha=0.15, linewidth=0.8)

    # Labels (No Title)
    plt.xlabel("Communication Rounds", fontsize=20, fontweight='bold')
    plt.ylabel("Training Loss", fontsize=20, fontweight='bold')
    
    # Spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2.0)
        
    ax.tick_params(which='major', width=2.0, length=6, labelsize=16)
    # Ensure tick labels are bold (needs trick or loop)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True, framealpha=1.0, 
               facecolor='white', edgecolor='#cccccc', borderpad=0.8)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved EN Loss Plot to {save_path}")

def plot_convergence_en(data_dict, save_path="results/figure/map_convergence.png"):
    ensure_dir(save_path)
    
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(12, 8))
    
    colors = sns.color_palette("tab10", n_colors=len(data_dict))
    idx = 0
    
    # 1. Upper Bound (Fixed 0.775)
    plt.axhline(y=0.7661, color='#ff6666', linestyle='--', linewidth=2.5, alpha=0.9, 
                label="Centralized (Upper Bound)")

    # 2. Experiments
    for name, df in data_dict.items():
        if "Centralized" in name: continue
        x_col = 'round' if 'round' in df.columns else 'epoch'
        color = colors[idx]; idx += 1
        
        plt.plot(df[x_col], df['mAP50'].rolling(5).mean(), label=name, color=color, linewidth=3.0)
        plt.plot(df[x_col], df['mAP50'], color=color, alpha=0.15, linewidth=1.0)

    plt.xlabel("Communication Rounds", fontsize=20, fontweight='bold')
    plt.ylabel("mAP@0.50", fontsize=20, fontweight='bold')
    
    # Spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2.0)
        
    ax.tick_params(which='major', width=2.0, length=6, labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.legend(loc='lower right', fontsize=14, frameon=True, fancybox=True, shadow=True, framealpha=1.0, 
               facecolor='white', edgecolor='#cccccc', borderpad=0.8)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved EN Convergence Plot to {save_path}")

def plot_efficiency_en(data_dict, save_path="results/figure/efficiency_tradeoff.png"):
    ensure_dir(save_path)
    
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(10, 8))
    
    points = []
    for name, df in data_dict.items():
        if "Centralized" in name: continue
        if 'mAP50' not in df.columns: continue
        
        if 'bits_up_compressed' in df.columns:
            total_gb = (df['bits_up_compressed'].sum() + df['bits_down_compressed'].sum()) / 8 / 1024**3
        else:
            model_size = 6.0 if "Nano" in name else 22.5
            if "8-bit" in name: model_size /= 4
            total_gb = (model_size * 2 * len(df)) / 1024
            
        points.append({'name': name, 'gb': total_gb, 'map': df['mAP50'].max()})
        
    for p in points:
        plt.scatter(p['gb'], p['map'], label=p['name'], s=250, alpha=0.9, edgecolors='white', linewidth=2)
        plt.text(p['gb'], p['map'] + 0.005, p['name'].split('(')[0], ha='center', va='bottom', fontsize=13, fontweight='bold')
        
    plt.xlabel("Total Communication Cost (GB)", fontsize=20, fontweight='bold')
    plt.ylabel("Best mAP50", fontsize=20, fontweight='bold')
    plt.xscale('log')
    
    # Spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2.0)
        
    ax.tick_params(which='major', width=2.0, length=6, labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.legend(loc='lower right', fontsize=14, frameon=True, fancybox=True, shadow=True, framealpha=1.0, 
               facecolor='white', edgecolor='#cccccc', borderpad=0.8)
    
    plt.grid(True, which="both", linestyle='--', alpha=0.4, color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved EN Efficiency Plot to {save_path}")

def generate_figure3(model_path, val_images_dir):
    # Visualization Placeholder
    pass 

# ================== Main ==================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='EN', choices=['EN', 'CN'])
    args = parser.parse_args()
    
    print(f"Generating plots for Language: {args.lang}")
    
    # 1. 加载数据
    if args.lang == "CN":
        experiments = {
            "集中式基准 (YOLOv8s)": "results/central_manual_3/central_log.csv",
            "FedAvg (YOLOv8s, 32位, 1 Epoch)": "results/runs_flq_v6_yolov8s_32bit_1epoch_back/experiment_data.csv",
            "本文方法 (YOLOv8s, 8位, 1 Epoch)": "results/runs_flq_v8_yolov8s_8bit_1epoch/experiment_data.csv",
        }
    else:
        experiments = {
            "Centralized (YOLOv8s)": "results/central_manual_v8s/central_log.csv",
            "FedAvg (YOLOv8s, 32-bit)": "results/runs_flq_v6_yolov8s_32bit_1epoch/experiment_data.csv",
            "FedAvg-Quant (YOLOv8s, 8-bit)": "results/runs_flq_v6_yolov8s_8bit_1epoch/experiment_data.csv",
            "Ours (YOLOv8n, 8-bit)": "results/runs_flq_v8_yolov8n_8bit_1epoch/experiment_data.csv",
        }
        
    data = load_data(experiments)
    
    if args.lang == "CN":
        plot_loss_cn(data)
        plot_convergence_cn(data)
        plot_efficiency_cn(data)
        # 生成中文版的 Cross 图
        plot_convergence_loss_cn(data)
    else:
        plot_loss_en(data)
        plot_convergence_en(data)
        plot_efficiency_en(data)
        # 生成英文版的 Cross 图
        plot_convergence_loss_en(data)
        
    print("Done.")

if __name__ == "__main__":
    main()
