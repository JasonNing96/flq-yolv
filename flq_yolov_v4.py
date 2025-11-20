#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLQ-YOLOv 联邦训练脚本（stable v5）

- 基于 Ultralytics YOLOv8；
- 单模型对象 + 外部全局 state_dict；
- 客户端本地训练使用全局权重初始化；
- 上行采用均匀量化（bits）模拟；
- 服务器端反量化 + FedAvg 聚合；
- 显式 DIL 链路扰动（2~10 Mbps, 0~20% 丢包）用于 bits_up/bits_down 统计。
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
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone().contiguous()
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ====================== 量化相关工具 ==========================

def quantize_tensor_uniform(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float]:
    x = x.detach().float()
    if bits >= 32:
        return x.clone(), 1.0
    qmax = 2 ** (bits - 1) - 1
    max_val = float(x.abs().max().item())
    if max_val < 1e-12:
        return torch.zeros_like(x, dtype=torch.int32), 1.0
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
    if bits >= 32:
        total_params = sum(v.numel() for v in q_delta.values()
                           if isinstance(v, torch.Tensor))
        return float(total_params * 32)

    total_params = sum(v.numel()
                       for v in q_delta.values() if isinstance(v, torch.Tensor))
    bits_weights = total_params * bits
    bits_scales = len(scales) * 32
    return float(bits_weights + bits_scales)


# ====================== DIL 链路建模 ==========================

def sample_dil_bandwidth() -> float:
    return random.uniform(2.0, 10.0)


def sample_dil_loss_ratio() -> float:
    return random.uniform(0.0, 0.2)


def apply_DIL_fluctuation(bits: float) -> float:
    bw_mbps = sample_dil_bandwidth()
    max_bits = bw_mbps * 1e6
    loss_ratio = sample_dil_loss_ratio()
    bits_after_loss = bits * (1.0 - loss_ratio)
    bits_limited = min(bits_after_loss, max_bits)
    return float(bits_limited)


# ====================== SD 差分 ==========================

def sd_delta(
    local_sd: Dict[str, torch.Tensor],
    global_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    delta: Dict[str, torch.Tensor] = {}
    for k, g in global_sd.items():
        if not isinstance(g, torch.Tensor):
            continue
        if k not in local_sd:
            delta[k] = torch.zeros_like(g, dtype=torch.float32)
            continue
        l = local_sd[k]
        if not isinstance(l, torch.Tensor) or l.shape != g.shape:
            delta[k] = torch.zeros_like(g, dtype=torch.float32)
            continue
        delta[k] = (l.detach().float() - g.detach().float())
    return delta


# ====================== 安全加载 state_dict（绕过 inference tensor） ==========================

def safe_load_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    """
    手动遍历 state_dict，将每个张量替换为新的 Tensor / Parameter，
    避免直接在“inference tensor”上做 inplace copy_ 引发错误。
    """
    with torch.no_grad():
        for name, tensor in sd.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            parts = name.split(".")
            obj = model
            for p in parts[:-1]:
                if hasattr(obj, p):
                    obj = getattr(obj, p)
                elif p.isdigit() and hasattr(obj, "__getitem__"):
                    obj = obj[int(p)]
                else:
                    obj = None
                    break
            if obj is None:
                continue
            attr = parts[-1]
            if not hasattr(obj, attr):
                continue
            cur = getattr(obj, attr)
            device = cur.device if isinstance(
                cur, torch.Tensor) else tensor.device
            new_tensor = tensor.to(device).clone()
            if isinstance(cur, nn.Parameter):
                setattr(obj, attr, nn.Parameter(new_tensor))
            else:
                setattr(obj, attr, new_tensor)


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
    以 global_sd 为初始化，在单个 client 上训练若干 epoch，返回本地 state_dict（CPU）。
    """
    # 下行：将全局参数载入当前 YOLO 模型
    safe_load_state_dict(model.model, global_sd)

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
        val=False,
        plots=False,
    )

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
    用当前全局模型在 val 集上评估一次，返回多个指标。
    """
    safe_load_state_dict(model.model, global_sd)

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

    precision = float(rd.get("metrics/precision(B)",
                      rd.get("metrics/precision", 0.0)))
    recall = float(rd.get("metrics/recall(B)", rd.get("metrics/recall", 0.0)))
    map50 = float(rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", 0.0)))
    map5095 = float(rd.get("metrics/mAP50-95(B)",
                    rd.get("metrics/mAP50-95", 0.0)))

    box_loss = float(
        rd.get(
            "val/box_loss",
            rd.get("train/box_loss", rd.get("box_loss", 0.0)),
        )
    )
    cls_loss = float(
        rd.get(
            "val/cls_loss",
            rd.get("train/cls_loss", rd.get("cls_loss", 0.0)),
        )
    )
    dfl_loss = float(
        rd.get(
            "val/dfl_loss",
            rd.get("train/dfl_loss", rd.get("dfl_loss", 0.0)),
        )
    )

    print(
        f"[Round {round_idx}] "
        f"P={precision:.4f}, R={recall:.4f}, "
        f"mAP50={map50:.4f}, mAP50-95={map5095:.4f}, "
        f"loss(box/cls/dfl)=({box_loss:.4f}/{cls_loss:.4f}/{dfl_loss:.4f})"
    )

    return {
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map5095,
        "box_loss": box_loss,
        "cls_loss": cls_loss,
        "dfl_loss": dfl_loss,
    }


# ====================== warmup：让 YOLO 头部结构匹配数据集 nc ==========================

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
    使用 example_yaml 跑一次 0-epoch 训练，使 YOLO 检测头结构适配数据集的类别数。
    """
    print(">>> warmup: 初始化 YOLO 并适配检测头结构...")
    model = YOLO(str(model_path))
    model.train(
        data=str(example_yaml),
        epochs=1,
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
    print(">>> warmup 完成。")
    return model, global_sd


# ====================== 主联邦循环 ==========================

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

    device = device if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")

    # warmup 得到已适配 2 类头部的 YOLO 和初始 global_sd
    yolo_model, global_sd = warmup_model_for_dataset(
        model_path=model_path,
        example_yaml=val_yaml,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        out_dir=out_dir,
    )

    num_params = sum(v.numel()
                     for v in global_sd.values() if isinstance(v, torch.Tensor))
    print("=" * 70)
    print("FLQ-YOLOv federated training (v5)")
    print(f"model: {model_path}")
    print(f"#clients: {len(client_yaml_list)}")
    print(f"bits (uplink): {bits}")
    print(f"#rounds: {rounds}, local_epochs: {local_epochs}")
    print(f"#params: {num_params:,}")
    print("=" * 70)

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

    for r in range(rounds):
        print(f"\n========== Round {r} / {rounds - 1} ==========")

        # 下行：全精度广播（统计用）
        bits_down_raw = num_params * 32
        bits_down = apply_DIL_fluctuation(bits_down_raw)
        print(
            f"[Round {r}] bits_down_raw ≈ {bits_down_raw / 1e6:.2f} Mbit, "
            f"after DIL ≈ {bits_down / 1e6:.2f} Mbit"
        )

        # 客户端训练
        q_deltas: List[Dict[str, torch.Tensor]] = []
        delta_scales: List[Dict[str, float]] = []
        bits_up_list: List[float] = []

        for i, yaml_path in enumerate(client_yaml_list):
            print(f"\n[Round {r}] training client #{i}: {yaml_path}")
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
                f"[Round {r}] client #{i} bits_up_raw ≈ {bits_up_raw / 1e6:.2f} Mbit, "
                f"after DIL ≈ {bits_up_real / 1e6:.2f} Mbit"
            )

        total_bits_up = sum(bits_up_list)
        print(
            f"\n[Round {r}] total bits_up ≈ {total_bits_up / 1e6:.2f} Mbit, "
            f"avg per client ≈ {total_bits_up / max(len(bits_up_list),1) / 1e6:.2f} Mbit"
        )

        # 服务器端聚合：反量化 + FedAvg
        print(f"[Round {r}] aggregating on server ...")
        agg_delta: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in global_sd.items()
            if isinstance(v, torch.Tensor)
        }

        for q_delta, scales in zip(q_deltas, delta_scales):
            deq_delta = dequantize_state_dict(q_delta, scales)
            for k in agg_delta.keys():
                if k in deq_delta and isinstance(deq_delta[k], torch.Tensor):
                    agg_delta[k] = agg_delta[k] + deq_delta[k].float()

        num_clients = max(len(client_yaml_list), 1)
        for k, v in global_sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            upd = agg_delta.get(k, torch.zeros_like(
                v, dtype=torch.float32)) / num_clients
            new_val = (v.detach().float() + upd).to(v.dtype)
            global_sd[k] = new_val.cpu().clone()

        # 使用最新 global_sd 在 val 集上评估一次
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

        # 记录日志
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

    torch.save(global_sd, out_dir / "global_last.pt")
    print(f"\n训练完成，最终模型保存在: {out_dir / 'global_last.pt'}")
    print(f"训练日志保存在: {out_dir / 'flq_log.json'}")


# ====================== CLI ==========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FLQ-YOLOv federated training (v5)")
    p.add_argument(
        "--clients",
        type=str,
        nargs="+",
        required=True,
        help="各客户端 data.yaml 路径列表",
    )
    p.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="全局验证 data.yaml 路径",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="YOLOv8 权重路径，例如 yolov8n.pt",
    )
    p.add_argument("--rounds", type=int, default=10, help="联邦训练轮数")
    p.add_argument("--local-epochs", type=int,
                   default=2, help="每轮每客户端本地 epochs")
    p.add_argument("--bits", type=int, default=8, help="上行更新量化比特数（1~32）")
    p.add_argument("--batch", type=int, default=16, help="训练/验证 batch size")
    p.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练使用的设备（如 '0' 或 'cpu'）",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader 的 num_workers（建议在本脚本中设为 0）",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="./runs_flq",
        help="输出目录",
    )
    return p.parse_args()


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
