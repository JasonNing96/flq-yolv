#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLQ-YOLOv 联邦训练（单机多 client 顺序模拟版）

特点：
- 全局模型 global_model 始终以 FP32 保存；
- 每个 client：以全局模型为初始，做若干本地 epoch 训练，得到本地模型；
- 用 state_dict 差分构造“更新向量”，按 bits 量化后再反量化；
- 多 client 的反量化更新做平均，加到全局模型上；
- 每轮用 val_data 对全局模型做验证，记录 mAP 与通信比特数。

用法示例（6 个客户端）：
python flq_yolov.py \
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
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ====================== 工具函数 ==========================

def seed_everything(seed: int = 42) -> None:
    """简单设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    """保证目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_sd_parameters(sd: Dict[str, torch.Tensor]) -> int:
    return sum(p.numel() for p in sd.values())


# def sd_clone_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#     return {k: v.detach().cpu().clone() for k, v in sd.items()}
def ensure_mutable_params(module: nn.Module) -> None:
    """
    将模块的所有参数从 inference tensor 变成普通可写 tensor。
    通过 .data.clone() 的方式重新绑定，避免 load_state_dict 时报
    'Inplace update to inference tensor outside InferenceMode is not allowed'.
    """
    with torch.no_grad():
        for p in module.parameters():
            if not isinstance(p, nn.Parameter):
                continue
            # 重新克隆一份数据并绑定回去
            new_data = p.data.clone().detach()
            p.data = new_data
            
def sd_clone_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    将 state_dict 彻底拷贝到 CPU，斩断梯度，避免 inference tensor。
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            # 非张量（例如某些 buffer）直接跳过或视情况原样返回
            continue
        out[k] = v.detach().cpu().clone().contiguous()
    return out



def sd_delta(local_sd: Dict[str, torch.Tensor],
             global_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """local - global，只处理形状匹配的 key"""
    delta = {}
    for k in global_sd.keys():
        if k not in local_sd:
            continue
        if not isinstance(global_sd[k], torch.Tensor) or not isinstance(
            local_sd[k], torch.Tensor
        ):
            continue
        if global_sd[k].shape != local_sd[k].shape:
            # 打印一下信息方便排查
            print(f"[warn] shape mismatch on key={k}, "
                  f"global={global_sd[k].shape}, local={local_sd[k].shape}")
            # 对于形状不匹配的，先记为 0（不会影响聚合）
            delta[k] = torch.zeros_like(global_sd[k], dtype=torch.float32)
            continue
        delta[k] = (local_sd[k].float() - global_sd[k].float())
    return delta


def quantize_tensor_uniform(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float]:
    """
    最简单的对称均匀量化（per-tensor）：
    - bits 包括符号位；例如 bits=8，则可表示 [-127, 127]
    - scale = max(|x|)/qmax
    - 返回 (q_int, scale)，其中 q_int 为 int8/int32 等整数张量
    """
    if bits <= 0 or bits >= 32:
        # 不量化，直接返回原张量（float32），scale=1.0
        return x.float(), 1.0

    qmax = 2 ** (bits - 1) - 1  # e.g., 8bit -> 127
    # 为了避免 scale 为 0，加一个极小值
    max_val = x.abs().max()
    if max_val == 0:
        return torch.zeros_like(x, dtype=torch.int32), 1.0

    scale = max_val / qmax
    q = torch.round(x / scale).clamp(-qmax - 1, qmax).to(torch.int32)
    return q, float(scale)


def dequantize_tensor_uniform(q: torch.Tensor, scale: float) -> torch.Tensor:
    """与 quantize_tensor_uniform 对应的反量化"""
    if not torch.is_tensor(q):
        raise TypeError("q must be a torch.Tensor")
    return q.float() * scale


def quantize_state_dict(
    delta_sd: Dict[str, torch.Tensor], bits: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    对 state_dict 差分做均匀量化（per-tensor，所有张量共享一个 bits）。
    返回：
    - q_sd: 每个 key 对应一个整数张量
    - scales: 每个 key 对应一个 scale（float）
    """
    q_sd = {}
    scales = {}
    for k, v in delta_sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        q, s = quantize_tensor_uniform(v, bits)
        q_sd[k] = q
        scales[k] = s
    return q_sd, scales


def dequantize_state_dict(
    q_sd: Dict[str, torch.Tensor],
    scales: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    将量化后的 state_dict 反量化回来（per-tensor）
    """
    deq = {}
    for k, q in q_sd.items():
        if k not in scales:
            continue
        deq[k] = dequantize_tensor_uniform(q, scales[k])
    return deq


def bits_for_quantized_delta(
    q_sd: Dict[str, torch.Tensor],
    bits: int,
    scales: Dict[str, float],
) -> float:
    """
    粗略统计本轮上传的 bit 数：
    - 假设每个整数元素占 bits 比特
    - 每个 tensor 有一个 scale，用 32 比特浮点表示
    """
    total_params = 0
    for k, q in q_sd.items():
        if not isinstance(q, torch.Tensor):
            continue
        total_params += q.numel()

    # weights 部分
    bits_weights = total_params * bits
    # 每个 tensor 一个 scale，按 32bit float 计
    bits_scales = len(scales) * 32
    return float(bits_weights + bits_scales)


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
    用某个 client 的数据，在 global_sd 初始化的模型上训练若干 epoch，
    返回本地训练后的 state_dict（CPU）。
    """
    # 1) 确保当前 YOLO 内部模型参数不是 inference tensor
    ensure_mutable_params(model.model)

    # 2) 取得当前模型参数所在 device（第一次一般是 cpu，之后是 cuda:0）
    try:
        param_device = next(model.model.parameters()).device
    except StopIteration:
        param_device = torch.device("cpu")

    # 3) 下行：根据当前 device 构造一份克隆后的全局权重
    global_sd_clone = {
        k: v.clone().to(param_device) for k, v in global_sd.items()
        if isinstance(v, torch.Tensor)
    }

    # 在 InferenceMode 下执行 load_state_dict，允许对 inference tensor 做就地更新
    with torch.inference_mode():
        model.model.load_state_dict(global_sd_clone, strict=False)

    # 4) 本地训练
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
        val=False,     # 本地训练不做额外验证
        plots=False,
    )

    # 5) 返回本地模型（训练后）的 CPU 版 state_dict
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
) -> Tuple[float, float]:
    """
    用全局模型在 val 数据集上做一次验证，返回 (mAP50, mAP50-95)
    """
    # 1) 确保模型参数不是 inference tensor
    ensure_mutable_params(model.model)

    # 2) 取得当前模型参数 device
    try:
        param_device = next(model.model.parameters()).device
    except StopIteration:
        param_device = torch.device("cpu")

    # 3) 克隆 global_sd，并搬到当前 device
    global_sd_clone = {
        k: v.clone().to(param_device) for k, v in global_sd.items()
        if isinstance(v, torch.Tensor)
    }

    # 在 InferenceMode 下执行 load_state_dict，避免对 inference tensor 的非法就地更新
    with torch.inference_mode():
        model.model.load_state_dict(global_sd_clone, strict=False)

    # 4) 验证阶段关闭梯度
    model.model.eval()
    with torch.no_grad():
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
    map50 = float(rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", 0.0)))
    map5095 = float(rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", 0.0)))
    return map50, map5095



# ====================== 主联邦训练流程 ==========================

def run_federated_flq(
    client_yaml_list: List[Path],
    val_yaml: Path,
    model_path: Path,
    out_dir: Path,
    rounds: int,
    local_epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    bits: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 YOLO 模型与全局参数
    # 首先读取第一个客户端的数据配置，确定类别数
    with open(client_yaml_list[0], "r", encoding="utf-8") as f:
        first_data_cfg = f.read()
    print(">>> 使用第一个客户端的数据配置：", client_yaml_list[0])

    # 初始化 YOLO 模型
    base_model = YOLO(str(model_path))
    print(f">>> Base model loaded from: {model_path}")
    print(f"    Total parameters: {count_parameters(base_model.model):,}")

    # 全局模型参数 state_dict（始终存 CPU 版本）
    global_sd = sd_clone_cpu(base_model.model.state_dict())

    # 日志
    log = {
        "round": [],
        "mAP50": [],
        "mAP50-95": [],
        "bits_up": [],
        "bits_down": [],
    }

    # device & workers 设置
    device = device if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")

    # 逐轮联邦训练
    for r in range(rounds):
        print(f"\n========== Round {r} / {rounds} ==========")

        # ---------------- 下行广播：将 global_sd 视作一次下行广播 ----------------
        # 简单起见：按所有可训练参数个数 * 32bit 计算一次广播的比特数
        num_params = count_sd_parameters(global_sd)
        bits_down = num_params * 32  # FP32
        print(f"[Round {r}] 下行广播 bits_down ≈ {bits_down / 1e6:.2f} Mbit")

        # ---------------- 各 client 本地训练 ----------------
        local_sds_cpu: List[Dict[str, torch.Tensor]] = []
        deltas: List[Dict[str, torch.Tensor]] = []
        bits_up_list: List[float] = []

        for i, yaml_path in enumerate(client_yaml_list):
            print(f"\n[Round {r}] 训练 client #{i}: {yaml_path}")

            # 复制一个 YOLO 对象，避免多个 client 共享同一内部状态
            client_model = YOLO(str(model_path))

            # 本地训练
            local_sd_cpu = train_one_client(
                model=client_model,
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
            local_sds_cpu.append(local_sd_cpu)

            # 计算 local - global 的差分
            delta = sd_delta(local_sd_cpu, global_sd)
            deltas.append(delta)

            # 做量化并计算上行 bit 数
            q_delta, scales = quantize_state_dict(delta, bits)
            bits_up = bits_for_quantized_delta(q_delta, bits, scales)
            bits_up_list.append(bits_up)
            print(
                f"[Round {r}] client #{i} 上行 bits_up ≈ "
                f"{bits_up / 1e6:.2f} Mbit (bits={bits})"
            )

            # 可以根据需要在这里直接存储当前轮 client 的量化结果、scale 等
            # 以便后续分析（本脚本只做粗略统计）

        # 汇总上行 bit 数
        total_bits_up = sum(bits_up_list)
        avg_bits_up = total_bits_up / len(client_yaml_list)
        print(
            f"\n[Round {r}] 总上行 bits_up ≈ {total_bits_up / 1e6:.2f} Mbit，"
            f"平均每 client ≈ {avg_bits_up / 1e6:.2f} Mbit"
        )

        # ---------------- 服务器端反量化 + 聚合 ----------------
        print(f"[Round {r}] 开始服务器端聚合...")

        # 初始化聚合累加器
        agg_delta: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in global_sd.items()
            if isinstance(v, torch.Tensor)
        }

        # 对每个 client 的 delta 做量化-反量化再聚合
        for idx, delta in enumerate(deltas):
            q_delta, scales = quantize_state_dict(delta, bits)
            deq_delta = dequantize_state_dict(q_delta, scales)

            for k in agg_delta.keys():
                if k in deq_delta and isinstance(deq_delta[k], torch.Tensor):
                    agg_delta[k] = agg_delta[k].float() + deq_delta[k].float()

        # 均值聚合后更新 global_sd
        num_clients = float(len(client_yaml_list))
        for k in agg_delta.keys():
            if k not in global_sd:
                continue
            if not isinstance(global_sd[k], torch.Tensor):
                continue
            new_val = (global_sd[k].float() + agg_delta[k] / num_clients).to(
                global_sd[k].dtype
            )
            global_sd[k] = new_val.clone()

        print(f"[Round {r}] 服务器端聚合完成。")

        # ---------------- 用最新 global_sd 做一次全局验证 ----------------
        print(f"[Round {r}] 开始全局验证...")
        eval_model = YOLO(str(model_path))  # 重新构造 YOLO
        map50, map5095 = evaluate_global(
            model=eval_model,
            global_sd=global_sd,
            val_yaml=val_yaml,
            round_idx=r,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
            out_dir=out_dir,
        )
        print(
            f"[Round {r}] mAP50={map50:.4f}, mAP50-95={map5095:.4f}, "
            f"bits_up={avg_bits_up / 1e6:.2f} Mbit, bits_down={bits_down / 1e6:.2f} Mbit"
        )

        # 记录日志
        log["round"].append(r)
        log["mAP50"].append(map50)
        log["mAP50-95"].append(map5095)
        log["bits_up"].append(avg_bits_up)
        log["bits_down"].append(bits_down)

        # 每轮保存当前 global_sd 以便中断恢复 / 分析
        torch.save(global_sd, out_dir / f"global_sd_round{r}.pt")

    # 训练结束，保存日志
    save_json(log, out_dir / "flq_log.json")
    print("\n========== 训练结束，日志已保存 ==========")

    # 做一个简单的 mAP 曲线图
    try:
        rounds_list = log["round"]
        plt.figure(figsize=(6, 4))
        plt.plot(rounds_list, log["mAP50"], label="mAP50")
        plt.plot(rounds_list, log["mAP50-95"], label="mAP50-95")
        plt.xlabel("Round")
        plt.ylabel("mAP")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig_path = out_dir / "mAP_curve.png"
        plt.savefig(fig_path, dpi=300)
        print(f"mAP 曲线已保存到: {fig_path}")
    except Exception as e:
        print(f"[warn] 绘制 mAP 曲线时出错: {e}")


# ====================== CLI 参数解析 ==========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FLQ-YOLOv: Simple Fed-Learning with Uniform Quantization"
    )
    p.add_argument(
        "--clients",
        type=str,
        nargs="+",
        required=True,
        help="各客户端 data.yaml 路径列表（例如 client1/oil.yaml client2/oil.yaml ...）",
    )
    p.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="全局验证用 data.yaml（若不指定，则使用第一个 client 的 data.yaml）",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="YOLOv 模型权重路径（例如 yolov8n.pt）",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="./runs_flq",
        help="输出目录",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="联邦训练轮数",
    )
    p.add_argument(
        "--local-epochs",
        type=int,
        default=2,
        help="每个客户端本地训练 epoch 数",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="训练和验证的 batch size",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="训练和验证的图像尺寸",
    )
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练使用的 device（例如 '0' 或 'cpu'）",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader 的 num_workers",
    )
    p.add_argument(
        "--bits",
        type=int,
        default=8,
        help="上行更新量化比特数（建议先用 8，32 表示不量化）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    client_yaml_list = [Path(p).resolve() for p in args.clients]
    val_yaml = Path(args.val_data).resolve() if args.val_data else client_yaml_list[0]
    model_path = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()

    seed_everything(42)

    run_federated_flq(
        client_yaml_list=client_yaml_list,
        val_yaml=val_yaml,
        model_path=model_path,
        out_dir=out_dir,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        bits=args.bits,
    )


if __name__ == "__main__":
    main()
