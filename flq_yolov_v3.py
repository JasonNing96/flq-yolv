#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLQ-YOLOv 联邦训练脚本（稳定版）

- 使用 Ultralytics YOLOv8 作为检测模型；
- 通过 state_dict 差分 + 低位宽（bits）量化模拟客户端上行更新；
- 在服务器端做反量化 + FedAvg 聚合；
- 显式模拟 DIL 链路（2~10 Mbps + 0~20% 丢包），记录 bits_up / bits_down；
- 每轮对全局模型做一次 val，记录 precision / recall / mAP50 / mAP50-95 以及 box/cls/dfl loss；
- 全过程仅使用单个 YOLO 对象，避免反复构图开销。

用法示例：

python flq_yolov_v4.py \
  --clients ./data/oil_detection_dataset/client1/oil.yaml \
           ./data/oil_detection_dataset/client2/oil.yaml \
           ./data/oil_detection_dataset/client3/oil.yaml \
           ./data/oil_detection_dataset/client4/oil.yaml \
           ./data/oil_detection_dataset/client5/oil.yaml \
           ./data/oil_detection_dataset/client6/oil.yaml \
  --val-data ./data/oil_detection_dataset/oil.yaml \
  --model ./models/yolov8n.pt \
  --rounds 10 \
  --local-epochs 2 \
  --bits 8
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO


# ====================== 杂项工具 ==========================

def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sd_clone_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    将 state_dict 彻底拷贝到 CPU，斩断梯度和计算图。
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone().contiguous()
    return out


def ensure_mutable_params(module: nn.Module) -> None:
    """
    防止参数成为 inference tensor：重新 clone 一份 data 绑定回去。
    """
    with torch.no_grad():
        for p in module.parameters():
            if isinstance(p, nn.Parameter):
                p.data = p.data.clone().detach().contiguous()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ====================== 量化相关工具 ==========================

def quantize_tensor_uniform(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float]:
    """
    对张量做对称均匀量化：
    - bits=32 时直接返回原值（视作 FP32，不做量化）；
    - 其它 bits 使用 [-max, max] 映射到整型区间。
    """
    x = x.detach().float()
    if bits >= 32:
        return x.clone(), 1.0

    qmax = 2 ** (bits - 1) - 1  # e.g. 127 for 8-bit
    max_val = float(x.abs().max().item())
    if max_val < 1e-12:
        return torch.zeros_like(x), 1.0

    scale = max_val / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int32)
    return q, float(scale)


def dequantize_tensor_uniform(q: torch.Tensor, scale: float) -> torch.Tensor:
    if q.dtype != torch.float32:
        q = q.to(torch.float32)
    return q * float(scale)


def quantize_state_dict(
    delta: Dict[str, torch.Tensor],
    bits: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    对 state_dict 形式的“梯度/更新”做逐层量化。
    返回量化后的 q_delta 以及对应的 scale 字典。
    """
    if bits >= 32:
        q_delta = {}
        scales = {}
        for k, v in delta.items():
            if isinstance(v, torch.Tensor):
                q_delta[k] = v.detach().clone().to(torch.float32)
                scales[k] = 1.0
        return q_delta, scales

    q_delta: Dict[str, torch.Tensor] = {}
    scales: Dict[str, float] = {}
    for k, v in delta.items():
        if not isinstance(v, torch.Tensor):
            continue
        q, s = quantize_tensor_uniform(v, bits)
        q_delta[k] = q
        scales[k] = s
    return q_delta, scales


def dequantize_state_dict(
    q_delta: Dict[str, torch.Tensor],
    scales: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, q in q_delta.items():
        if k not in scales:
            continue
        out[k] = dequantize_tensor_uniform(q, scales[k])
    return out


def bits_for_quantized_delta(
    q_delta: Dict[str, torch.Tensor],
    bits: int,
    scales: Dict[str, float],
) -> float:
    """
    估算一个量化更新需要的比特数：
    - 每个量化权重 bits；
    - 每个 scale 以 FP32 发送（32 bits）。
    """
    if bits >= 32:
        # 视作 FP32，全精度上行
        total_params = sum(v.numel() for v in q_delta.values() if isinstance(v, torch.Tensor))
        return float(total_params * 32)

    total_params = sum(v.numel() for v in q_delta.values() if isinstance(v, torch.Tensor))
    bits_weights = total_params * bits
    bits_scales = len(scales) * 32
    return float(bits_weights + bits_scales)


# ====================== DIL 链路建模 ==========================

def sample_dil_bandwidth() -> float:
    """采样 2~10 Mbps 之间的链路带宽。"""
    return random.uniform(2.0, 10.0)


def sample_dil_loss_ratio() -> float:
    """采样 0~20% 丢包率。"""
    return random.uniform(0.0, 0.2)


def apply_DIL_fluctuation(bits: float) -> float:
    """
    对“理论 bits”加入 DIL 约束：带宽限制 + 丢包。
    这里只做简单缩放，用于统计；
    真实时间建模可以在论文中单独给出。
    """
    bw_mbps = sample_dil_bandwidth()
    max_bits = bw_mbps * 1e6  # 假设 1s 传输时间
    loss_ratio = sample_dil_loss_ratio()

    bits_after_loss = bits * (1.0 - loss_ratio)
    bits_limited = min(bits_after_loss, max_bits)
    return float(bits_limited)


# ====================== SD 差分 ==========================

def sd_delta(
    local_sd: Dict[str, torch.Tensor],
    global_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    计算 local - global，仅在 key 存在且形状一致时参与更新。
    """
    delta: Dict[str, torch.Tensor] = {}
    for k, g in global_sd.items():
        if not isinstance(g, torch.Tensor):
            continue
        if k not in local_sd:
            # 缺失 key，则视为 0 更新
            delta[k] = torch.zeros_like(g, dtype=torch.float32)
            continue
        l = local_sd[k]
        if not isinstance(l, torch.Tensor) or l.shape != g.shape:
            # 理论上 warmup 后不会出现 shape mismatch，这里做保护
            delta[k] = torch.zeros_like(g, dtype=torch.float32)
            continue
        delta[k] = (l.detach().float() - g.detach().float())
    return delta


# ====================== 训练与评估 ==========================

def train_one_client(
    model: YOLO,
    global_sd: Dict[str, torch.Tensor],
    data_yaml: Path,
    round_idx: int,
    client_idx: int,
    local_epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    out_dir: Path,
) -> Dict[str, torch.Tensor]:
    """
    在给定 client 数据上，以 global_sd 为初始化做若干 epoch 训练。
    返回该 client 训练结束后的 state_dict（CPU 版）。
    """
    # 下行：加载全局参数到当前设备
    ensure_mutable_params(model.model)
    try:
        param_device = next(model.model.parameters()).device
    except StopIteration:
        param_device = torch.device("cpu")

    global_sd_clone = {
        k: v.clone().to(param_device)
        for k, v in global_sd.items()
        if isinstance(v, torch.Tensor)
    }
    with torch.no_grad():
        model.model.load_state_dict(global_sd_clone, strict=False)

    run_name = f"round{round_idx}_client{client_idx}"
    model.train(
        data=str(data_yaml),
        epochs=local_epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=str(out_dir / "train"),
        name=run_name,
        exist_ok=True,
        verbose=False,
        val=False,   # 本地训练阶段不单独做 val
        plots=False,
    )

    # 返回 CPU 版参数
    return sd_clone_cpu(model.model.state_dict())


def evaluate_global(
    model: YOLO,
    global_sd: Dict[str, torch.Tensor],
    val_yaml: Path,
    round_idx: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    out_dir: Path,
) -> Dict[str, float]:
    """
    用当前全局模型在 val 集上做一次评估，返回包含多种指标的字典。
    """
    # 将 global_sd 加载到模型
    ensure_mutable_params(model.model)
    try:
        param_device = next(model.model.parameters()).device
    except StopIteration:
        param_device = torch.device("cpu")

    global_sd_clone = {
        k: v.clone().to(param_device)
        for k, v in global_sd.items()
        if isinstance(v, torch.Tensor)
    }
    with torch.no_grad():
        model.model.load_state_dict(global_sd_clone, strict=False)

    results = model.val(
        data=str(val_yaml),
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=str(out_dir / "val"),
        name=f"round{round_idx}",
        exist_ok=True,
        verbose=False,
        plots=False,
    )

    rd = getattr(results, "results_dict", {}) or {}

    # 兼容不同版本 ultralytics 的 key
    precision = float(rd.get("metrics/precision(B)",
                             rd.get("metrics/precision", 0.0)))
    recall = float(rd.get("metrics/recall(B)",
                          rd.get("metrics/recall", 0.0)))
    map50 = float(rd.get("metrics/mAP50(B)",
                         rd.get("metrics/mAP50", 0.0)))
    map5095 = float(rd.get("metrics/mAP50-95(B)",
                           rd.get("metrics/mAP50-95", 0.0)))

    box_loss = float(rd.get("val/box_loss",
                            rd.get("train/box_loss",
                                   rd.get("box_loss", 0.0))))
    cls_loss = float(rd.get("val/cls_loss",
                            rd.get("train/cls_loss",
                                   rd.get("cls_loss", 0.0))))
    dfl_loss = float(rd.get("val/dfl_loss",
                            rd.get("train/dfl_loss",
                                   rd.get("dfl_loss", 0.0))))

    metrics = {
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map5095,
        "box_loss": box_loss,
        "cls_loss": cls_loss,
        "dfl_loss": dfl_loss,
    }

    print(
        f"[Round {round_idx}] "
        f"P={precision:.4f}, R={recall:.4f}, "
        f"mAP50={map50:.4f}, mAP50-95={map5095:.4f}, "
        f"loss(box/cls/dfl)=({box_loss:.4f}/{cls_loss:.4f}/{dfl_loss:.4f})"
    )

    return metrics


# ====================== 主联邦训练流程 ==========================

def warmup_model_for_dataset(
    model_path: Path,
    example_yaml: Path,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    out_dir: Path,
) -> Tuple[YOLO, Dict[str, torch.Tensor]]:
    """
    使用一个 example yaml（通常是全局 val.yaml），
    做一次 0-epoch train，将 YOLO 的检测头结构适配到正确的 nc（类别数），
    然后导出作为全局初始模型。
    """
    print(">>> 初始化 YOLO 模型并根据数据集适配检测头结构 (warmup)...")
    model = YOLO(str(model_path))

    # 0 个 epoch，主要目的是触发内部的数据集解析与 nc 适配逻辑
    model.train(
        data=str(example_yaml),
        epochs=0,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=str(out_dir / "warmup"),
        name="warmup",
        exist_ok=True,
        verbose=False,
        val=False,
        plots=False,
    )

    global_sd = sd_clone_cpu(model.model.state_dict())
    print(">>> warmup 完成，检测头已适配到数据集类别数。")
    return model, global_sd


def run_federated_flq(
    client_yaml_list: List[Path],
    val_yaml: Path,
    model_path: Path,
    rounds: int,
    local_epochs: int,
    bits: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    out_dir: Path,
) -> None:
    seed_everything(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 初始化 YOLO 模型 & 全局参数 ----------
    device = device if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")

    yolo_model, global_sd = warmup_model_for_dataset(
        model_path=model_path,
        example_yaml=val_yaml,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        out_dir=out_dir,
    )

    # 日志结构：每轮一条
    log = {
        "round": [],
        "precision": [],
        "recall": [],
        "mAP50": [],
        "mAP50-95": [],
        "box_loss": [],
        "cls_loss": [],
        "dfl_loss": [],
        "bits_up": [],
        "bits_down": [],
    }

    print("=" * 70)
    print("FLQ-YOLOv 联邦训练（稳定版）")
    print(f"模型: {model_path}")
    print(f"客户端数: {len(client_yaml_list)}")
    print(f"量化比特数 (bits): {bits}")
    print(f"联邦轮数: {rounds}, 每轮本地 epochs: {local_epochs}")
    print("=" * 70)

    num_params = sum(v.numel() for v in global_sd.values()
                     if isinstance(v, torch.Tensor))

    for r in range(rounds):
        print(f"\n========== Round {r} / {rounds - 1} ==========")

        # ---------- 下行广播 bits 统计 ----------
        bits_down_raw = num_params * 32  # 仍按全精度下行
        bits_down = apply_DIL_fluctuation(bits_down_raw)
        print(
            f"[Round {r}] 下行 bits_down_raw ≈ {bits_down_raw / 1e6:.2f} Mbit, "
            f"after DIL ≈ {bits_down / 1e6:.2f} Mbit"
        )

        # ---------- 各 client 本地训练 ----------
        q_deltas: List[Dict[str, torch.Tensor]] = []
        delta_scales: List[Dict[str, float]] = []
        bits_up_list: List[float] = []

        for i, yaml_path in enumerate(client_yaml_list):
            print(f"\n[Round {r}] 训练 client #{i}: {yaml_path}")

            local_sd_cpu = train_one_client(
                model=yolo_model,
                global_sd=global_sd,
                data_yaml=yaml_path,
                round_idx=r,
                client_idx=i,
                local_epochs=local_epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                workers=workers,
                out_dir=out_dir,
            )

            delta = sd_delta(local_sd_cpu, global_sd)
            q_delta, scales = quantize_state_dict(delta, bits)
            q_deltas.append(q_delta)
            delta_scales.append(scales)

            bits_up_raw = bits_for_quantized_delta(q_delta, bits, scales)
            bits_up_real = apply_DIL_fluctuation(bits_up_raw)
            bits_up_list.append(bits_up_real)

            print(
                f"[Round {r}] client #{i} bits_up_raw ≈ "
                f"{bits_up_raw / 1e6:.2f} Mbit, after DIL ≈ {bits_up_real / 1e6:.2f} Mbit"
            )

        total_bits_up = sum(bits_up_list)
        avg_bits_up = total_bits_up / max(len(bits_up_list), 1)
        print(
            f"\n[Round {r}] 总上行 bits_up ≈ {total_bits_up / 1e6:.2f} Mbit, "
            f"平均每 client ≈ {avg_bits_up / 1e6:.2f} Mbit"
        )

        # ---------- 服务器端聚合（只反量化一次） ----------
        print(f"[Round {r}] 开始服务器端聚合...")

        agg_delta: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in global_sd.items()
            if isinstance(v, torch.Tensor)
        }

        for q_delta, scales in zip(q_deltas, delta_scales):
            deq_delta = dequantize_state_dict(q_delta, scales)
            for k in agg_delta.keys():
                if k in deq_delta and isinstance(deq_delta[k], torch.Tensor):
                    agg_delta[k] = agg_delta[k].float() + deq_delta[k].float()

        num_clients = max(len(client_yaml_list), 1)
        for k in global_sd.keys():
            v = global_sd[k]
            if not isinstance(v, torch.Tensor):
                continue
            upd = agg_delta.get(k, torch.zeros_like(v, dtype=torch.float32)) / num_clients
            new_val = (v.detach().float() + upd).to(v.dtype)
            global_sd[k] = new_val.cpu().clone()

        # ---------- 用最新全局模型在 val 集上评估 ----------
        metrics = evaluate_global(
            model=yolo_model,
            global_sd=global_sd,
            val_yaml=val_yaml,
            round_idx=r,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
            out_dir=out_dir,
        )

        # ---------- 记录日志 ----------
        log["round"].append(r)
        log["precision"].append(metrics["precision"])
        log["recall"].append(metrics["recall"])
        log["mAP50"].append(metrics["mAP50"])
        log["mAP50-95"].append(metrics["mAP50-95"])
        log["box_loss"].append(metrics["box_loss"])
        log["cls_loss"].append(metrics["cls_loss"])
        log["dfl_loss"].append(metrics["dfl_loss"])
        log["bits_up"].append(total_bits_up)
        log["bits_down"].append(bits_down)

        save_json(log, out_dir / "flq_log.json")
        torch.save(global_sd, out_dir / f"global_round_{r}.pt")

    # 最终模型
    torch.save(global_sd, out_dir / "global_last.pt")
    print(f"\n训练完成，最终全局模型已保存到: {out_dir / 'global_last.pt'}")
    print(f"日志已保存到: {out_dir / 'flq_log.json'}")


# ====================== CLI 入口 ==========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLQ-YOLOv federated training")
    parser.add_argument(
        "--clients",
        nargs="+",
        type=str,
        required=True,
        help="多个 client 对应的数据 yaml 路径",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="全局验证使用的数据 yaml 路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="YOLOv8 模型权重路径，如 yolov8n.pt",
    )
    parser.add_argument("--rounds", type=int, default=10, help="联邦训练轮数")
    parser.add_argument("--local-epochs", type=int, default=2, help="每轮每个 client 的本地 epochs")
    parser.add_argument("--bits", type=int, default=8, help="量化比特数（1/2/4/8/16/32）")
    parser.add_argument("--batch", type=int, default=16, help="本地训练 batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练/验证使用的设备（例如 '0' 或 'cpu'）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="dataloader workers 数量，建议在本脚本中设为 0 以避免多进程问题",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./runs_flq",
        help="联邦训练输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client_yaml_list = [Path(p) for p in args.clients]
    val_yaml = Path(args.val_data)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)

    run_federated_flq(
        client_yaml_list=client_yaml_list,
        val_yaml=val_yaml,
        model_path=model_path,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        bits=args.bits,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
