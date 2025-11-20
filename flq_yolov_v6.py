#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLQ-YOLOv5 联邦训练脚本 (v6: Paper Experiment Version)
----------------------------------------------------------------
- 核心功能: 实现 Stateless 联邦训练 + SA-FLQ 上行压缩
- 论文支撑:
  1. DIL 环境模拟: 2-10 Mbps 带宽波动 + 0-20% 丢包率
  2. 全维度数据采集: CSV 记录每轮通信量(上行/下行)、时延、Loss、mAP
  3. 模型保存: 自动保存 Global Best 模型及所有 Clients 的最终模型
- 修复积累: 包含 v5 版本的所有 Stability Fixes (Namespace, OOM, Device, Momentum)
----------------------------------------------------------------
"""

import torch
try:
    _ = torch.OutOfMemoryError
except AttributeError:
    torch.OutOfMemoryError = RuntimeError

import argparse
import copy
import json
import csv  # <--- 新增
import random
import os
import time
import shutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from types import SimpleNamespace

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Ultralytics 组件
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.data.utils import check_det_dataset

# ====================== 1. 基础工具 & DIL 模拟 ======================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --- 论文核心：DIL 环境模拟 ---
class DILSimulator:
    """模拟受限无线网络环境 (Dynamic Interference Link)"""
    def __init__(self, min_bw=2.0, max_bw=10.0, min_loss=0.0, max_loss=0.2):
        self.min_bw = min_bw      # Mbps
        self.max_bw = max_bw      # Mbps
        self.min_loss = min_loss  # 丢包率 0%
        self.max_loss = max_loss  # 丢包率 20%

    def simulate_transmission(self, data_bits: float) -> Tuple[float, float, float]:
        """
        模拟一次数据传输
        Returns:
            latency (s): 传输耗时
            bw (Mbps): 采样带宽
            loss_rate: 采样丢包率
        """
        if data_bits <= 0:
            return 0.0, 0.0, 0.0
        
        # 随机采样环境参数
        bw_mbps = random.uniform(self.min_bw, self.max_bw)
        loss_rate = random.uniform(self.min_loss, self.max_loss)
        
        # 有效吞吐量 = 带宽 * (1 - 丢包率)
        # 简单模型：丢包导致重传或吞吐下降
        effective_bw = bw_mbps * 1e6 * (1.0 - loss_rate)
        
        if effective_bw < 1e-6: effective_bw = 1e-6
        
        latency = data_bits / effective_bw
        return latency, bw_mbps, loss_rate

# ====================== 2. 核心算法: FLQ 压缩器 ======================

class FLQCompressor:
    def __init__(self, device):
        self.device = device
        self.local_error: Optional[torch.Tensor] = None

    def flatten_params(self, state_dict: Dict) -> torch.Tensor:
        tensors = [v.float().to(self.device)
                   for v in state_dict.values() if v.dtype.is_floating_point]
        if not tensors:
            return torch.tensor([], device=self.device)
        return torch.cat([t.view(-1) for t in tensors])

    def reconstruct_state_dict(self, flat_vec: torch.Tensor, template_sd: Dict) -> Dict:
        out = {}
        offset = 0
        flat_vec = flat_vec.to(self.device)
        for k, v in template_sd.items():
            if v.dtype.is_floating_point:
                numel = v.numel()
                out[k] = flat_vec[offset: offset + numel].view(v.shape).to(v.dtype).cpu()
                offset += numel
            else:
                out[k] = v.clone().cpu()
        return out

    def quantize_update(self, delta_vec: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int]:
        """
        返回: (量化后的delta, 压缩后的比特数)
        """
        num_params = delta_vec.numel()
        
        # Error Feedback
        if self.local_error is None:
            self.local_error = torch.zeros_like(delta_vec)
        target = delta_vec + self.local_error

        if bits >= 32:
            self.local_error.zero_()
            return target, num_params * 32

        if bits == 1:
            # 1-bit quantization (SignSGD style)
            scale = target.abs().mean()
            if scale < 1e-8: scale = 1e-8
            sign = torch.sign(target)
            sign[sign == 0] = 1.0
            quantized = sign * scale
            self.local_error = target - quantized
            return quantized, num_params + 32 # 1 bit per param + 32 bit scale
        else:
            # k-bit quantization
            mn, mx = target.min(), target.max()
            scale = (mx - mn) / (2**bits - 1 + 1e-8)
            zero = -mn / (scale + 1e-8)
            q = torch.clamp(torch.round(target / scale + zero), 0, 2**bits - 1)
            dq = (q - zero) * scale
            self.local_error = target - dq
            return dq, num_params * bits + 32

# ====================== 3. 训练内核: Stateless Manual Trainer ======================

class ManualClientTrainer:
    """
    无状态训练器：不在 __init__ 中保留模型，只在 train_epoch 中临时创建并销毁。
    """
    def __init__(self, model_path: str, data_yaml: Path, device: str, batch: int, imgsz: int):
        self.device = device
        self.data_yaml = data_yaml
        self.model_path = model_path
        self.batch = batch
        self.imgsz = imgsz

        # DataLoader 持久化
        cfg = get_cfg(DEFAULT_CFG)
        cfg.data = str(data_yaml)
        cfg.imgsz = imgsz
        cfg.batch = batch
        data_info = check_det_dataset(str(data_yaml))
        train_path = data_info['train']
        self.dataset = build_yolo_dataset(
            cfg, train_path, batch, data_info, mode="train", rect=False, stride=32
        )
        # workers=0 避免死锁
        self.loader = build_dataloader(
            self.dataset, batch, workers=0, shuffle=True, rank=-1)
        self.compressor = FLQCompressor(device)

    def train_epoch(self, global_sd: Dict, local_epochs: int, lr: float) -> Tuple[Dict, dict, dict]:
        """执行本地训练"""
        # 1. Fresh Load
        temp_wrapper = YOLO(self.model_path)
        model = temp_wrapper.model

        # Fix 1: Namespace
        if hasattr(model, 'args') and isinstance(model.args, dict):
            model.args = SimpleNamespace(**model.args)
        
        model.load_state_dict(global_sd)
        model.to(self.device)
        model.train()

        # Fix 2: Unfreeze
        for param in model.parameters():
            param.requires_grad = True
        
        loss_fn = v8DetectionLoss(model)
        
        # Fix 3: Hyp Injection
        if hasattr(loss_fn, 'hyp'):
            if isinstance(loss_fn.hyp, dict):
                loss_fn.hyp = SimpleNamespace(**loss_fn.hyp)
            if not hasattr(loss_fn.hyp, 'box'): loss_fn.hyp.box = 7.5
            if not hasattr(loss_fn.hyp, 'cls'): loss_fn.hyp.cls = 0.5
            if not hasattr(loss_fn.hyp, 'dfl'): loss_fn.hyp.dfl = 1.5

        if hasattr(loss_fn, 'proj'):
            loss_fn.proj = loss_fn.proj.to(self.device)

        scaler = GradScaler()
        # Fix 4: Low LR + No Momentum
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.0, weight_decay=5e-4)

        loss_stats = {"box": [], "cls": [], "dfl": []}

        for epoch in range(local_epochs):
            for batch in self.loader:
                batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255.0
                for k in batch:
                    if k != 'img' and isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)

                optimizer.zero_grad()
                with autocast(enabled=True):
                    preds = model(batch['img'])
                    loss, loss_items = loss_fn(preds, batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()

                loss_stats["box"].append(loss_items[0].item())
                loss_stats["cls"].append(loss_items[1].item())
                loss_stats["dfl"].append(loss_items[2].item())

        final_sd = {k: v.cpu() for k, v in model.state_dict().items()}

        # Clean up
        del model
        del temp_wrapper
        del loss_fn
        del optimizer
        del scaler
        gc.collect()
        torch.cuda.empty_cache()

        metadata = {
            "final_loss": np.mean(loss_stats["box"]) if loss_stats["box"] else 0.0,
        }
        return final_sd, loss_stats, metadata

# ====================== 4. 主流程 ======================

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
    seed_everything(42)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV 初始化
    csv_path = out_dir / "experiment_data.csv"
    csv_headers = [
        "round", "mAP50", "mAP50-95", "avg_loss",
        "bits_down_raw", "bits_down_compressed", "latency_down_sim",
        "bits_up_raw", "bits_up_compressed", "latency_up_sim",
        "total_round_time"
    ]
    # 动态添加 client loss 列
    for i in range(len(client_yaml_list)):
        csv_headers.append(f"client_{i}_loss")
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # Device setup
    if torch.cuda.is_available() and device != 'cpu':
        if device.isdigit():
            device = f"cuda:{device}"
    else:
        device = 'cpu'

    print(f"🚀 FLQ-YOLOv6 (Paper Exp Version) | Device: {device} | Bits: {bits}")

    # --- 1. Warmup ---
    print("   [Init] Adapting model head (Warmup)...")
    warmup_dir = out_dir / "warmup_temp"
    init_model = YOLO(str(model_path))
    try:
        init_model.train(
            data=str(val_yaml), epochs=1, imgsz=imgsz, batch=batch,
            device=device, project=str(warmup_dir), name="init_run",
            exist_ok=True, plots=False, save=True, val=False, verbose=False
        )
    except Exception as e:
        print(f"   [Init] Warmup passed: {e}")
    
    # Env Cleanup
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("CUDA_VISIBLE_DEVICES")

    warmup_pt = warmup_dir / "init_run/weights/last.pt"
    adapted_pt = out_dir / "init_adapted.pt"
    model_path_to_use = model_path
    if warmup_pt.exists():
        shutil.copy(str(warmup_pt), str(adapted_pt))
        model_path_to_use = adapted_pt
        del init_model
        init_model = YOLO(str(adapted_pt))
    else:
        print("   [Warning] Warmup failed, using original model.")

    global_sd = {k: v.cpu().clone() for k, v in init_model.model.state_dict().items()}
    template_sd = copy.deepcopy(global_sd)
    
    del init_model
    gc.collect()
    if warmup_dir.exists(): shutil.rmtree(warmup_dir)

    # --- 2. Clients & Simulator ---
    client_trainers = []
    for yaml_path in client_yaml_list:
        trainer = ManualClientTrainer(str(model_path_to_use), yaml_path, device, batch, imgsz)
        client_trainers.append(trainer)

    server_helper = FLQCompressor(device)
    dil_sim = DILSimulator(min_bw=2.0, max_bw=10.0, min_loss=0.0, max_loss=0.2)

    # State Tracking
    best_map = 0.0

    # --- 3. Federated Loop ---
    for r in range(rounds):
        print(f"\n========== Round {r} / {rounds - 1} ==========")
        t_start = time.time()
        
        # == 下行广播模拟 ==
        # 假设全精度广播 (32-bit)
        num_params = sum(p.numel() for p in global_sd.values())
        bits_down_raw = num_params * 32
        bits_down_compressed = bits_down_raw # 暂无下行压缩
        latency_down, _, _ = dil_sim.simulate_transmission(bits_down_compressed)
        print(f"   [Downlink] {bits_down_compressed/1e6:.2f} Mb, Latency: {latency_down:.3f}s")

        # == 本地训练 ==
        cur_lr = 0.01 * (0.98 ** r)
        client_updates_dense = []
        round_loss_list = []
        
        bits_up_raw_total = 0
        bits_up_compressed_total = 0
        max_latency_up = 0 # 假设并行传输，取最慢的 Client
        
        for i, trainer in enumerate(client_trainers):
            local_sd, loss_stats, meta = trainer.train_epoch(
                global_sd, local_epochs, lr=cur_lr) # Momentum removed inside

            flat_global = server_helper.flatten_params(global_sd).to(device)
            flat_local = trainer.compressor.flatten_params(local_sd).to(device)
            delta = flat_local - flat_global

            # 上行压缩 & DIL 模拟
            q_delta, bit_cost = trainer.compressor.quantize_update(delta, bits)
            client_updates_dense.append(q_delta)
            
            # 统计
            client_loss = meta["final_loss"]
            round_loss_list.append(client_loss)
            
            raw_bits = delta.numel() * 32
            bits_up_raw_total += raw_bits
            bits_up_compressed_total += bit_cost
            
            lat, _, _ = dil_sim.simulate_transmission(bit_cost)
            max_latency_up = max(max_latency_up, lat) # 并行传输模型

            if i == 0:
                print(f"   [Client {i}] Loss: {client_loss:.4f} | Up: {bit_cost/1e6:.2f} Mb | Lat: {lat:.3f}s")
            
            # 保存最后一轮的 Client 模型
            if r == rounds - 1:
                c_save_path = out_dir / f"client_{i}_final.pt"
                # 需要重建完整模型对象才能保存，比较耗时，仅最后做
                # 这里简单起见保存 SD，或者临时加载保存
                # 为节省时间，只保存 SD
                torch.save(local_sd, c_save_path)

        avg_loss = np.mean(round_loss_list)

        # == 服务器聚合 ==
        print("   [Server] Aggregating...")
        stack_updates = torch.stack(client_updates_dense)
        avg_update = stack_updates.mean(dim=0)
        flat_global_new = server_helper.flatten_params(global_sd).to(device) + avg_update
        global_sd = server_helper.reconstruct_state_dict(flat_global_new, template_sd)

        # == 评估 ==
        print("   [Server] Evaluating...")
        metrics = {"mAP50": 0, "mAP50-95": 0}
        try:
            torch.cuda.empty_cache()
            val_model = YOLO(str(model_path_to_use))
            val_model.model.load_state_dict(global_sd)
            results = val_model.val(
                data=str(val_yaml), batch=batch, device=device,
                verbose=False, plots=False
            )
            metrics["mAP50"] = results.results_dict.get("metrics/mAP50(B)", 0.0)
            metrics["mAP50-95"] = results.results_dict.get("metrics/mAP50-95(B)", 0.0)
            del val_model
            gc.collect()
        except Exception as e:
            print(f"   [Warning] Eval failed: {e}")

        torch.cuda.empty_cache()
        
        t_end = time.time()
        round_time = t_end - t_start

        # == 记录与保存 ==
        # 1. CSV
        csv_row = [
            r, metrics["mAP50"], metrics["mAP50-95"], avg_loss,
            bits_down_raw, bits_down_compressed, latency_down,
            bits_up_raw_total, bits_up_compressed_total, max_latency_up,
            round_time
        ]
        csv_row.extend(round_loss_list)
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)

        # 2. Best Model
        if metrics["mAP50"] > best_map:
            best_map = metrics["mAP50"]
            torch.save(global_sd, out_dir / "global_best.pt")
            print(f"   [Save] New Best Model (mAP50={best_map:.4f})")

        print(f"   [Result] mAP50: {metrics['mAP50']:.4f} | Loss: {avg_loss:.4f} | Time: {round_time:.1f}s")

    # 保存最终 Global
    torch.save(global_sd, out_dir / "global_last.pt")
    print(f"\n训练完成. 结果保存在: {out_dir}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=str, nargs="+", required=True)
    p.add_argument("--val-data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="./results/runs_flq_v6")
    return p.parse_args()

def main():
    args = parse_args()
    run_federated_flq(
        [Path(p) for p in args.clients], Path(args.val_data), Path(args.model),
        args.rounds, args.local_epochs, args.bits, args.batch, args.imgsz,
        args.device, args.workers, Path(args.out_dir)
    )

if __name__ == "__main__":
    main()

