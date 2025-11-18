#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal FLQ-YOLOv training script (single-machine, single-client).

- 不依赖 server/client 进程，只在本地模拟“下行模型 + 上行梯度”的 FLQ 过程；
- 每一轮：
  1) 对全局模型做一次下行量化（可选），记录 bits_down；
  2) 在本地数据上训练若干 epoch，得到本地模型；
  3) 计算本地模型与全局模型的差分向量，做上行量化（可选），记录 bits_up；
  4) 在“服务器侧”反量化并更新全局模型；
  5) 记录 mAP50 / Loss / bits_{up,down} 等到 CSV，并可画图。

示例：
    python flq_yolov.py --data data/client1/oil.yaml --model yolov8n.pt --rounds 5 --local-epochs 1
"""

import argparse
import csv
import logging

from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

import torch
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ==================== 量化与向量工具（摘自 model_utils.py，稍作整理） ====================

def state_dict_to_vector(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """将 state_dict 展平为一维向量"""
    return torch.cat([v.view(-1) for v in state_dict.values()])


def vector_to_state_dict(vector: torch.Tensor, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """按 template 形状将一维向量还原为 state_dict"""
    state_dict: Dict[str, torch.Tensor] = {}
    pointer = 0
    for key, value in template.items():
        num_param = value.numel()
        state_dict[key] = vector[pointer:pointer + num_param].view_as(value)
        pointer += num_param
    return state_dict


def state_dict_to_grad_vector(client_state_dict: Dict[str, torch.Tensor],
                              global_state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """client 与 global 的差分向量 (client - global)，展平为一维向量"""
    grads = []
    # 只处理两个 state_dict 都存在的 key
    common_keys = set(client_state_dict.keys()) & set(global_state_dict.keys())
    if len(common_keys) != len(global_state_dict):
        missing = set(global_state_dict.keys()) - common_keys
        extra = set(client_state_dict.keys()) - common_keys
        if missing:
            logging.warning(f"global_state_dict 中有 {len(missing)} 个 key 在 client_state_dict 中不存在: {list(missing)[:5]}...")
        if extra:
            logging.warning(f"client_state_dict 中有 {len(extra)} 个 key 在 global_state_dict 中不存在: {list(extra)[:5]}...")
    
    # 按照 global_state_dict 的顺序处理
    for key in global_state_dict.keys():
        if key not in client_state_dict:
            # 如果 client 中没有这个 key，使用 global 的值（梯度为0）
            global_tensor = global_state_dict[key]
            grads.append(torch.zeros_like(global_tensor).view(-1))
            continue
            
        client_tensor = client_state_dict[key]
        global_tensor = global_state_dict[key]
        
        # 检查形状是否匹配
        if client_tensor.shape != global_tensor.shape:
            logging.warning(f"Key {key} 形状不匹配: client {client_tensor.shape} vs global {global_tensor.shape}, 使用零梯度")
            grads.append(torch.zeros_like(global_tensor).view(-1))
            continue
        
        # 确保两个 tensor 在同一设备上
        if client_tensor.device != global_tensor.device:
            global_tensor = global_tensor.to(client_tensor.device)
        
        # 检查 nan/inf
        diff = client_tensor - global_tensor
        if torch.isnan(diff).any() or torch.isinf(diff).any():
            logging.warning(f"Key {key} 的梯度包含 nan/inf，使用零梯度")
            grads.append(torch.zeros_like(global_tensor).view(-1))
        else:
            grads.append(diff.view(-1))
    
    return torch.cat(grads)


def grad_vector_to_state_dict(grad_vector: torch.Tensor,
                              global_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """将差分向量应用到 global_state_dict 上，得到新的 state_dict"""
    new_state: Dict[str, torch.Tensor] = {}
    pointer = 0
    for key, value in global_state_dict.items():
        num_param = value.numel()
        grad_param = grad_vector[pointer:pointer + num_param].view_as(value)
        new_state[key] = value + grad_param
        pointer += num_param
    return new_state


def quantize_vector(vector: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
    """统一量化到给定 bit 数，返回 (量化后的整数向量, scale, zero_point)."""
    if bits == 32:  # 视作 FP32，不做量化
        return vector, 1.0, 0.0

    v = vector.detach().cpu()  # 确保在 CPU 上，避免设备冲突
    
    # 检查并处理 nan/inf 值
    if torch.isnan(v).any() or torch.isinf(v).any():
        # 将 nan 和 inf 替换为 0
        v = torch.where(torch.isnan(v) | torch.isinf(v), torch.zeros_like(v), v)
    
    min_val = v.min().item()
    max_val = v.max().item()
    
    # 检查 nan/inf
    if min_val != min_val or max_val != max_val or not (min_val <= max_val):
        return torch.zeros_like(v), 0.0, 0.0

    if bits == 1:
        # 符号量化：{-1, +1} * scale
        scale = max(abs(min_val), abs(max_val))
        if scale == 0.0 or scale != scale:  # 检查 nan
            return torch.zeros_like(v), 0.0, 0.0
        q = torch.sign(v)
        zero_point = 0.0
    else:
        levels = 2 ** bits
        scale = (max_val - min_val) / (levels - 1) if max_val > min_val else 0.0
        zero_point = min_val
        if scale == 0.0 or scale != scale:  # 检查 nan
            return torch.zeros_like(v), 0.0, 0.0
        q = torch.round((v - zero_point) / scale)
        q = torch.clamp(q, 0, levels - 1)

    return q, float(scale), float(zero_point)


def dequantize_vector(quantized: torch.Tensor, scale: float, zero_point: float, bits: int = 8) -> torch.Tensor:
    """反量化一维向量"""
    if bits == 32:
        return quantized
    if bits == 1:
        return quantized * scale
    return quantized * scale + zero_point


def compute_model_size(state_dict: Dict[str, torch.Tensor], bits: int = 32) -> Tuple[int, float]:
    """计算参数量与模型大小 (MB)"""
    num_params = sum(p.numel() for p in state_dict.values())
    size_bytes = num_params * bits / 8
    size_mb = size_bytes / (1024 ** 2)
    return num_params, float(size_mb)


def compute_compression_ratio(original_bits: int, compressed_bits: int) -> float:
    if compressed_bits <= 0:
        return 1.0
    return original_bits / compressed_bits


def clone_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """克隆 state_dict，确保在 CPU 上（避免设备冲突）"""
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


# ==================== 绘图与 CSV ====================

def save_metrics_csv(rows: List[Dict], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_curves(rows: List[Dict], out_dir: Path) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = [r["round"] for r in rows]
    map50 = [r.get("map50", 0.0) for r in rows]
    map5095 = [r.get("map5095", 0.0) for r in rows]
    bits_up = [r.get("bits_up", 0.0) for r in rows]
    bits_down = [r.get("bits_down", 0.0) for r in rows]
    bits_up_cum = [r.get("bits_up_cum", 0.0) for r in rows]
    bits_down_cum = [r.get("bits_down_cum", 0.0) for r in rows]

    # 1) Round - mAP50 / mAP50-95
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, map50, marker="o", label="mAP50")
    if any(v > 0 for v in map5095):
        plt.plot(rounds, map5095, marker="s", label="mAP50-95")
    plt.xlabel("Round")
    plt.ylabel("mAP")
    plt.grid(True)
    plt.legend()
    plt.title("mAP vs Round (FLQ-YOLOv)")
    plt.tight_layout()
    plt.savefig(out_dir / "map_vs_round.png")
    plt.close()

    # 2) Round - bits_up / bits_down
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, bits_up, marker="o", label="bits_up")
    plt.plot(rounds, bits_down, marker="s", label="bits_down")
    plt.xlabel("Round")
    plt.ylabel("Bits per round")
    plt.grid(True)
    plt.legend()
    plt.title("Communication Bits per Round (FLQ-YOLOv)")
    plt.tight_layout()
    plt.savefig(out_dir / "bits_vs_round.png")
    plt.close()

    # 3) Cumulative bits - mAP50
    plt.figure(figsize=(8, 5))
    plt.plot(bits_up_cum, map50, marker="o", label="mAP50 vs cum bits_up")
    plt.xlabel("Cumulative uplink bits")
    plt.ylabel("mAP50")
    plt.grid(True)
    plt.legend()
    plt.title("mAP50 vs Cumulative Uplink Bits (FLQ-YOLOv)")
    plt.tight_layout()
    plt.savefig(out_dir / "map_vs_cum_bits_up.png")
    plt.close()


# ==================== 主训练逻辑 ====================

def run_flq_yolov(
    data_yaml: Path,
    model_path: Path,
    out_dir: Path,
    rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "cuda",
    workers: int = 0,
    quant_bits: int = 1,
    down_bits: int = 0,
    verbose: bool = True,
    enable_val: bool = True,
    enable_plots: bool = False,
    no_plot: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 加载数据配置，确定类别数
    with data_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    nc = int(data_cfg.get("nc", 80))

    # 初始化 YOLO 模型
    model = YOLO(str(model_path))
    from ultralytics.nn.tasks import DetectionModel
    model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)

    # 初始全局模型
    global_state = clone_state_dict(model.model.state_dict())
    num_params, model_size_mb = compute_model_size(global_state, bits=32)

    logger.info("=" * 70)
    logger.info("FLQ-YOLOv 单机训练")
    logger.info("=" * 70)
    logger.info(f"数据: {data_yaml}")
    logger.info(f"模型: {model_path}")
    logger.info(f"类别数 nc: {nc}")
    logger.info(f"参数量: {num_params:,} ({model_size_mb:.2f} MB @ 32-bit)")
    logger.info(f"轮次: {rounds}, 每轮本地 epoch: {local_epochs}")
    logger.info(f"量化: uplink={quant_bits} bit, downlink={down_bits if down_bits > 0 else 32} bit")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 70)

    # 统计
    rows: List[Dict] = []
    bits_up_cum = 0.0
    bits_down_cum = 0.0

    for r in range(rounds):
        logger.info(f"\n{'='*70}")
        logger.info(f"Round {r} / {rounds - 1}")
        logger.info(f"{'='*70}")
        print(f"\n----- Round {r} / {rounds - 1} -----")

        # === 下行：对全局模型做量化（可选） ===
        global_vec = state_dict_to_vector(global_state)
        
        # 检查全局模型参数是否有效
        if torch.isnan(global_vec).any() or torch.isinf(global_vec).any():
            logger.error(f"Round {r}: 全局模型参数包含 nan/inf，跳过量化，使用 FP32")
            down_state = global_state
            bits_down = int(global_vec.numel()) * 32
        elif down_bits and down_bits != 32:
            q_down, s_d, zp_d = quantize_vector(global_vec, bits=down_bits)
            down_vec = dequantize_vector(q_down, s_d, zp_d, bits=down_bits)
            
            # 检查量化后的参数是否有效
            if torch.isnan(down_vec).any() or torch.isinf(down_vec).any():
                logger.warning(f"Round {r}: 量化后参数包含 nan/inf，回退到 FP32")
                down_state = global_state
                bits_down = int(global_vec.numel()) * 32
            else:
                down_state = vector_to_state_dict(down_vec, global_state)
                bits_down = int(global_vec.numel()) * int(down_bits)
        else:
            down_state = global_state
            bits_down = int(global_vec.numel()) * 32

        bits_down_cum += bits_down

        # 将下行模型加载到本地 YOLO
        try:
            missing_keys, unexpected_keys = model.model.load_state_dict(down_state, strict=False)
            if missing_keys:
                logger.warning(f"Round {r}: 加载模型时缺少 {len(missing_keys)} 个 key: {missing_keys[:3]}...")
            if unexpected_keys:
                logger.warning(f"Round {r}: 加载模型时有 {len(unexpected_keys)} 个意外的 key: {unexpected_keys[:3]}...")
            
            # 检查加载后的模型参数是否有效
            model.model.eval()
            with torch.no_grad():
                test_state = model.model.state_dict()
                test_vec = state_dict_to_vector(test_state)
                if torch.isnan(test_vec).any() or torch.isinf(test_vec).any():
                    logger.error(f"Round {r}: 加载后的模型参数包含 nan/inf，使用上一轮的全局模型")
                    # 回退到上一轮的全局模型
                    if r > 0:
                        model.model.load_state_dict(global_state, strict=False)
                    else:
                        # Round 0 时重新加载预训练模型
                        model = YOLO(str(model_path))
                        from ultralytics.nn.tasks import DetectionModel
                        model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)
                        down_state = clone_state_dict(model.model.state_dict())
            model.model.train()
        except Exception as e:
            logger.error(f"Round {r}: 加载模型失败: {e}")
            if r > 0:
                logger.info(f"Round {r}: 回退到上一轮的全局模型")
                model.model.load_state_dict(global_state, strict=False)
                down_state = global_state
            else:
                raise

        # === 本地训练 ===
        results = model.train(
            data=str(data_yaml),
            epochs=local_epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=workers,
            project=str(out_dir),
            name=f"round_{r}",
            exist_ok=True,
            verbose=verbose,
            val=enable_val,
            plots=enable_plots,
        )

        # 提取关键指标
        metrics = {}
        
        # 方法1: 优先从训练历史CSV文件读取（YOLO会保存训练历史，最可靠）
        csv_path = out_dir / f"round_{r}" / "results.csv"
        if csv_path.exists():
            try:
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    csv_rows = list(reader)
                    if csv_rows:
                        last_row = csv_rows[-1]
                        # 遍历所有列，匹配指标
                        for key, value in last_row.items():
                            key_lower = key.lower()
                            try:
                                val = float(value) if value else 0.0
                                # 检查是否为有效数值（非 nan）
                                if val != val:  # nan check
                                    continue
                                
                                if "map50" in key_lower and "95" not in key_lower and "map50-95" not in key_lower:
                                    if metrics.get("map50", 0.0) == 0.0 or val > 0:
                                        metrics["map50"] = val
                                elif ("map50-95" in key_lower or "map50_95" in key_lower) and metrics.get("map5095", 0.0) == 0.0:
                                    metrics["map5095"] = val
                                elif "precision" in key_lower and "/" not in key and metrics.get("precision", 0.0) == 0.0:
                                    metrics["precision"] = val
                                elif "recall" in key_lower and "/" not in key and metrics.get("recall", 0.0) == 0.0:
                                    metrics["recall"] = val
                                elif "box_loss" in key_lower:
                                    metrics["loss"] = metrics.get("loss", 0.0) + val
                                elif "cls_loss" in key_lower:
                                    metrics["loss"] = metrics.get("loss", 0.0) + val
                                elif "dfl_loss" in key_lower:
                                    metrics["loss"] = metrics.get("loss", 0.0) + val
                            except (ValueError, TypeError):
                                continue
                        logger.info(f"从CSV读取指标: {csv_path}")
            except Exception as e:
                logger.warning(f"无法从CSV读取指标: {e}")
        
        # 方法2: 从 results_dict 提取（作为补充）
        if (metrics.get("map50", 0.0) == 0.0 or metrics.get("loss", 0.0) == 0.0) and hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            rd = results.results_dict
            # 尝试多种可能的key名称
            if metrics.get("map50", 0.0) == 0.0:
                metrics["map50"] = float(rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", rd.get("map50", 0.0))))
            if metrics.get("map5095", 0.0) == 0.0:
                metrics["map5095"] = float(rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", rd.get("map50-95", rd.get("map", 0.0)))))
            if metrics.get("precision", 0.0) == 0.0:
                metrics["precision"] = float(rd.get("metrics/precision(B)", rd.get("metrics/precision", rd.get("precision", 0.0))))
            if metrics.get("recall", 0.0) == 0.0:
                metrics["recall"] = float(rd.get("metrics/recall(B)", rd.get("metrics/recall", rd.get("recall", 0.0))))
            if metrics.get("loss", 0.0) == 0.0:
                metrics["loss"] = float(
                    rd.get("train/box_loss", 0.0)
                    + rd.get("train/cls_loss", 0.0)
                    + rd.get("train/dfl_loss", 0.0)
                )
        
        # 方法3: 从训练器的metrics属性提取（作为补充）
        if hasattr(results, "trainer"):
            trainer = results.trainer
            # 尝试从trainer的metrics获取
            if hasattr(trainer, "metrics") and isinstance(trainer.metrics, dict):
                m = trainer.metrics
                if metrics.get("map50", 0.0) == 0.0:
                    metrics["map50"] = float(m.get("metrics/mAP50(B)", m.get("map50", m.get("metrics/mAP50", 0.0))))
                if metrics.get("map5095", 0.0) == 0.0:
                    metrics["map5095"] = float(m.get("metrics/mAP50-95(B)", m.get("map50-95", m.get("map", m.get("metrics/mAP50-95", 0.0)))))
                if metrics.get("precision", 0.0) == 0.0:
                    metrics["precision"] = float(m.get("metrics/precision(B)", m.get("precision", m.get("metrics/precision", 0.0))))
                if metrics.get("recall", 0.0) == 0.0:
                    metrics["recall"] = float(m.get("metrics/recall(B)", m.get("recall", m.get("metrics/recall", 0.0))))
                if metrics.get("loss", 0.0) == 0.0:
                    metrics["loss"] = float(
                        m.get("train/box_loss", 0.0)
                        + m.get("train/cls_loss", 0.0)
                        + m.get("train/dfl_loss", 0.0)
                    )
            
            # 尝试从trainer的last属性获取（训练历史）
            if hasattr(trainer, "last") and isinstance(trainer.last, dict):
                last = trainer.last
                if metrics.get("loss", 0.0) == 0.0:
                    metrics["loss"] = float(
                        last.get("train/box_loss", 0.0)
                        + last.get("train/cls_loss", 0.0)
                        + last.get("train/dfl_loss", 0.0)
                    )
        
        # 如果仍然没有获取到，设置默认值
        if metrics.get("map50", 0.0) == 0.0 and metrics.get("loss", 0.0) == 0.0:
            metrics.setdefault("loss", 0.0)
            metrics.setdefault("map50", 0.0)
            metrics.setdefault("map5095", 0.0)
            metrics.setdefault("precision", 0.0)
            metrics.setdefault("recall", 0.0)
            
            # 调试：打印results对象的结构
            if verbose and r == 0:
                print(f"\n[调试] results对象属性: {dir(results)}")
                if hasattr(results, "results_dict"):
                    print(f"[调试] results_dict keys: {list(results.results_dict.keys()) if isinstance(results.results_dict, dict) else 'N/A'}")
                if hasattr(results, "trainer"):
                    print(f"[调试] trainer.metrics: {getattr(results.trainer, 'metrics', 'N/A')}")

        # 方法4: 如果仍然没有获取到验证指标，手动运行验证
        if metrics.get("map50", 0.0) == 0.0 and enable_val:
            try:
                # 使用训练后的模型进行验证
                val_results = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
                if hasattr(val_results, "results_dict") and isinstance(val_results.results_dict, dict):
                    vd = val_results.results_dict
                    if metrics.get("map50", 0.0) == 0.0:
                        metrics["map50"] = float(vd.get("metrics/mAP50(B)", vd.get("metrics/mAP50", vd.get("map50", 0.0))))
                    if metrics.get("map5095", 0.0) == 0.0:
                        metrics["map5095"] = float(vd.get("metrics/mAP50-95(B)", vd.get("metrics/mAP50-95", vd.get("map50-95", vd.get("map", 0.0)))))
                    if metrics.get("precision", 0.0) == 0.0:
                        metrics["precision"] = float(vd.get("metrics/precision(B)", vd.get("metrics/precision", vd.get("precision", 0.0))))
                    if metrics.get("recall", 0.0) == 0.0:
                        metrics["recall"] = float(vd.get("metrics/recall(B)", vd.get("metrics/recall", vd.get("recall", 0.0))))
            except Exception as e:
                if verbose:
                    print(f"警告: 手动验证失败: {e}")
        
        # 检查并处理 nan 值
        for key in ["map50", "map5095", "precision", "recall", "loss"]:
            val = metrics.get(key, 0.0)
            if val != val or not isinstance(val, (int, float)):  # nan check
                metrics[key] = 0.0
                logger.warning(f"Round {r}: {key} 为 nan，已重置为 0.0")
        
        logger.info(
            f"Round {r} 训练完成: mAP50={metrics['map50']:.4f}, mAP50-95={metrics['map5095']:.4f}, "
            f"precision={metrics.get('precision', 0.0):.4f}, recall={metrics.get('recall', 0.0):.4f}, "
            f"loss={metrics['loss']:.4f}"
        )
        print(
            f"本轮训练完成: mAP50={metrics['map50']:.4f}, mAP50-95={metrics['map5095']:.4f}, "
            f"precision={metrics.get('precision', 0.0):.4f}, recall={metrics.get('recall', 0.0):.4f}, "
            f"loss={metrics['loss']:.4f}"
        )

        # === 计算差分向量并做上行量化 ===
        local_state = clone_state_dict(model.model.state_dict())
        
        # 检查本地模型参数是否有效
        local_vec = state_dict_to_vector(local_state)
        if torch.isnan(local_vec).any() or torch.isinf(local_vec).any():
            logger.error(f"Round {r}: 本地模型参数包含 nan/inf，无法计算梯度，跳过本轮更新")
            # 使用零梯度
            grad_vec = torch.zeros_like(state_dict_to_vector(down_state))
        else:
            try:
                grad_vec = state_dict_to_grad_vector(local_state, down_state)
                
                # 检查梯度是否有效
                if torch.isnan(grad_vec).any() or torch.isinf(grad_vec).any():
                    logger.warning(f"Round {r}: 梯度包含 nan/inf，将 nan/inf 替换为 0")
                    grad_vec = torch.where(torch.isnan(grad_vec) | torch.isinf(grad_vec), 
                                          torch.zeros_like(grad_vec), grad_vec)
            except KeyError as e:
                logger.error(f"Round {r}: 计算梯度时出现 KeyError: {e}")
                logger.error(f"local_state keys: {list(local_state.keys())[:5]}...")
                logger.error(f"down_state keys: {list(down_state.keys())[:5]}...")
                # 使用零梯度
                grad_vec = torch.zeros_like(state_dict_to_vector(down_state))

        if quant_bits and quant_bits != 32:
            q_up, s_u, zp_u = quantize_vector(grad_vec, bits=quant_bits)
            bits_up = int(grad_vec.numel()) * int(quant_bits)
            deq_grad = dequantize_vector(q_up, s_u, zp_u, bits=quant_bits)
        else:
            deq_grad = grad_vec
            bits_up = int(grad_vec.numel()) * 32

        bits_up_cum += bits_up

        # === “服务器侧”聚合（单 client 情况下就是自己） ===
        new_global_state = grad_vector_to_state_dict(deq_grad, down_state)
        global_state = clone_state_dict(new_global_state)

        compress_ratio = compute_compression_ratio(32, quant_bits if quant_bits else 32)

        row = dict(
            round=r,
            map50=metrics.get("map50", 0.0),
            map5095=metrics.get("map5095", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            loss=metrics.get("loss", 0.0),
            bits_up=bits_up,
            bits_down=bits_down,
            bits_up_cum=bits_up_cum,
            bits_down_cum=bits_down_cum,
            compress_ratio=compress_ratio,
        )
        rows.append(row)

        logger.info(
            f"Round {r} 通信: bits_up={bits_up/1e6:.2f} Mbit, bits_down={bits_down/1e6:.2f} Mbit, "
            f"累计上传={bits_up_cum/1e6:.2f} Mbit, 累计下载={bits_down_cum/1e6:.2f} Mbit"
        )
        print(
            f"通信: bits_up={bits_up/1e6:.2f} Mbit, bits_down={bits_down/1e6:.2f} Mbit, 累计上传={bits_up_cum/1e6:.2f} Mbit"
        )

    # 保存最终全局模型与指标
    ckpt_path = out_dir / "flq_yolov_global_last.pt"
    torch.save(global_state, ckpt_path)
    logger.info(f"最终全局模型已保存到: {ckpt_path}")
    print(f"\n最终全局模型已保存到: {ckpt_path}")

    csv_path = out_dir / "flq_yolov_metrics.csv"
    save_metrics_csv(rows, csv_path)
    logger.info(f"训练指标已保存到: {csv_path}")
    print(f"训练指标已保存到: {csv_path}")

    if not no_plot:
        plot_dir = out_dir / "plots"
        plot_curves(rows, plot_dir)
        logger.info(f"曲线图已保存到: {plot_dir}")
        print(f"曲线图已保存到: {plot_dir}")
    
    logger.info("=" * 70)
    logger.info("训练完成！")
    logger.info("=" * 70)


# ==================== 命令行入口 ====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("最小 FLQ-YOLOv 训练脚本 (单机)")
    parser.add_argument("--data", type=str, default="data/client1/oil.yaml", help="YOLO 数据集 YAML 路径")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 权重路径 (例如 yolov8n.pt)")
    parser.add_argument("--out-dir", type=str, default="outputs/flq_yolov", help="输出目录")

    parser.add_argument("--rounds", type=int, default=5, help="联邦轮次 (global rounds)")
    parser.add_argument("--local-epochs", type=int, default=1, help="每轮本地训练 epoch 数")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="输入分辨率")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备，如 'cuda' 或 'cpu'")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers 数，建议 0")

    parser.add_argument("--quant-bits", type=int, default=1, help="上行梯度量化比特数 (1/4/8/32)")
    parser.add_argument("--down-bits", type=int, default=0, help="下行模型量化比特数，0/32 表示 FP32")

    parser.add_argument("--no-plot", action="store_true", help="只训练，不画收敛与通信曲线")
    parser.add_argument("--no-val", action="store_true", help="关闭 YOLO 验证阶段 (提升一点速度)")
    parser.add_argument("--plots", action="store_true", help="保存 YOLO 内部的训练曲线图")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = project_root / data_yaml

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    run_flq_yolov(
        data_yaml=data_yaml,
        model_path=model_path,
        out_dir=out_dir,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        quant_bits=args.quant_bits,
        down_bits=args.down_bits,
        verbose=True,
        enable_val=not args.no_val,
        enable_plots=args.plots,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
