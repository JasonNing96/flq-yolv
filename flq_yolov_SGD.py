#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLQ-YOLOv8 SGD Control Group (Ordinary Quantization, No Error Feedback)
----------------------------------------------------------------
- Core: Standard FedAvg with Ordinary Quantization (No Error Feedback)
- Purpose: Baseline comparison for FLQ (which uses Error Feedback)
- Based on: FLQ-YOLOv8 (FreezeBN Version)
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
import csv
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

class DILSimulator:
    """模拟受限无线网络环境 (Dynamic Interference Link)"""
    def __init__(self, min_bw=2.0, max_bw=10.0, min_loss=0.0, max_loss=0.2):
        self.min_bw = min_bw      # Mbps
        self.max_bw = max_bw      # Mbps
        self.min_loss = min_loss  # 丢包率 0%
        self.max_loss = max_loss  # 丢包率 20%

    def simulate_transmission(self, data_bits: float) -> Tuple[float, float, float]:
        if data_bits <= 0:
            return 0.0, 0.0, 0.0
        
        bw_mbps = random.uniform(self.min_bw, self.max_bw)
        loss_rate = random.uniform(self.min_loss, self.max_loss)
        
        effective_bw = bw_mbps * 1e6 * (1.0 - loss_rate)
        if effective_bw < 1e-6: effective_bw = 1e-6
        
        latency = data_bits / effective_bw
        return latency, bw_mbps, loss_rate

# ====================== 2. 核心算法: FLQ 压缩器 ======================

class FLQCompressor:
    def __init__(self, device):
        self.device = device
        self.local_error: Optional[torch.Tensor] = None
    
    def reset_error(self):
        """重置 error feedback，释放显存"""
        if self.local_error is not None:
            del self.local_error
            self.local_error = None
            torch.cuda.empty_cache()
    
    def get_error_state(self) -> Optional[torch.Tensor]:
        """获取 error feedback 状态（用于 checkpoint）"""
        if self.local_error is not None:
            return self.local_error.cpu().clone()
        return None
    
    def set_error_state(self, error_state: Optional[torch.Tensor]):
        """恢复 error feedback 状态（用于 checkpoint）"""
        if error_state is not None:
            self.local_error = error_state.to(self.device)
        else:
            self.local_error = None

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
        [SGD Control Group] Ordinary Quantization (No Error Feedback)
        Directly quantize the update vector without adding/saving residual error.
        """
        num_params = delta_vec.numel()
        
        # 1. No Error Feedback: target is just the current delta
        target = delta_vec

        if bits >= 32:
            return target, num_params * 32

        # 2. Ordinary Min-Max Quantization
        mn, mx = target.min(), target.max()
        
        # Safety check
        if torch.isnan(mn) or torch.isnan(mx) or (mx - mn) == 0:
            scale = 1.0
            zero = 0.0
        else:
            scale = (mx - mn) / (2**bits - 1 + 1e-8)
            zero = -mn / (scale + 1e-8)
        
        q = torch.clamp(torch.round(target / scale + zero), 0, 2**bits - 1)
        dq = (q - zero) * scale
        
        # 3. No Error Saving
        self.local_error = None
        
        return dq, num_params * bits + 32

# ====================== 3. 训练内核: Stateless Manual Trainer ======================

class ManualClientTrainer:
    def __init__(self, model_path: str, data_yaml: Path, device: str, batch: int, imgsz: int):
        self.device = device
        self.data_yaml = data_yaml
        self.model_path = model_path
        self.batch = batch
        self.imgsz = imgsz

        self.cfg = get_cfg(DEFAULT_CFG)
        self.cfg.data = str(data_yaml)
        self.cfg.imgsz = imgsz
        self.cfg.batch = batch
        self.data_info = check_det_dataset(str(data_yaml))
        self.train_path = self.data_info['train']
        self.batch_size = batch
        self.compressor = FLQCompressor(device)

    def train_epoch(self, global_sd: Dict, local_epochs: int, lr: float) -> Tuple[Dict, dict, dict]:
        temp_wrapper = YOLO(self.model_path)
        model = temp_wrapper.model
        del temp_wrapper  # 立即释放 wrapper
        torch.cuda.empty_cache()

        if hasattr(model, 'args') and isinstance(model.args, dict):
            model.args = SimpleNamespace(**model.args)
        
        model.load_state_dict(global_sd)
        model.to(self.device)
        
        # ================== FREEZE BN LOGIC ==================
        # 1. Set train mode for everything first
        model.train()
        # 2. Iterate and freeze BN layers (turn into eval mode)
        #    This stops them from updating running_mean/running_var
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False # Critical: Stop tracking stats
                module.eval() # Freeze behavior
        # =====================================================

        for param in model.parameters():
            param.requires_grad = True
        
        loss_fn = v8DetectionLoss(model)
        
        if hasattr(loss_fn, 'hyp'):
            if isinstance(loss_fn.hyp, dict):
                loss_fn.hyp = SimpleNamespace(**loss_fn.hyp)
            if not hasattr(loss_fn.hyp, 'box'): loss_fn.hyp.box = 7.5
            if not hasattr(loss_fn.hyp, 'cls'): loss_fn.hyp.cls = 0.5
            if not hasattr(loss_fn.hyp, 'dfl'): loss_fn.hyp.dfl = 1.5

        if hasattr(loss_fn, 'proj'):
            loss_fn.proj = loss_fn.proj.to(self.device)

        scaler = GradScaler()
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.0, weight_decay=5e-4)

        # 每次训练时重新创建 DataLoader（避免持久化占用显存）
        dataset = build_yolo_dataset(
            self.cfg, self.train_path, self.batch_size, self.data_info, mode="train", rect=False, stride=32
        )
        loader = build_dataloader(
            dataset, self.batch_size, workers=0, shuffle=True, rank=-1)

        loss_stats = {"box": [], "cls": [], "dfl": []}

        for epoch in range(local_epochs):
            for batch_idx, batch in enumerate(loader):
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
                
                # 及时释放 batch 和 preds 的显存引用
                del preds, loss, loss_items
                for k in list(batch.keys()):
                    if isinstance(batch[k], torch.Tensor):
                        del batch[k]
                del batch
                
                # 每 100 个 batch 清理一次显存
                if (batch_idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()

        final_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Clean up - 更彻底的清理
        optimizer.zero_grad(set_to_none=True)
        del model
        del loss_fn
        del optimizer
        del scaler
        del loader
        del dataset
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        metadata = {
            "final_loss": np.mean(loss_stats["box"]) if loss_stats["box"] else 0.0,
        }
        return final_sd, loss_stats, metadata

# ====================== 4. Checkpoint 管理 ======================

def save_checkpoint(
    global_sd: Dict,
    template_sd: Dict,
    client_trainers: List[ManualClientTrainer],
    round_num: int,
    best_map: float,
    out_dir: Path,
    device: str
) -> None:
    """保存 checkpoint"""
    checkpoint = {
        'round': round_num,
        'global_state_dict': global_sd,
        'template_state_dict': template_sd,
        'best_map': best_map,
        'client_errors': []
    }
    
    for trainer in client_trainers:
        error_state = trainer.compressor.get_error_state()
        checkpoint['client_errors'].append(error_state)
    
    ckpt_path = out_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"   [Checkpoint] Saved to {ckpt_path}")
    
    if (round_num + 1) % 10 == 0:
        ckpt_path_hist = out_dir / f"checkpoint_round_{round_num+1}.pt"
        torch.save(checkpoint, ckpt_path_hist)
        print(f"   [Checkpoint] Saved history checkpoint to {ckpt_path_hist}")

def load_checkpoint(
    checkpoint_path: Path,
    client_trainers: List[ManualClientTrainer],
    device: str
) -> Tuple[Dict, Dict, int, float]:
    """加载 checkpoint"""
    if not checkpoint_path.exists():
        return None, None, 0, 0.0
    
    print(f"   [Checkpoint] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    global_sd = checkpoint['global_state_dict']
    template_sd = checkpoint['template_state_dict']
    start_round = checkpoint['round'] + 1
    best_map = checkpoint.get('best_map', 0.0)
    
    client_errors = checkpoint.get('client_errors', [])
    for i, trainer in enumerate(client_trainers):
        if i < len(client_errors):
            trainer.compressor.set_error_state(client_errors[i])
    
    print(f"   [Checkpoint] Resumed from round {checkpoint['round']}, best_map={best_map:.4f}")
    return global_sd, template_sd, start_round, best_map

# ====================== 5. 主流程 ======================

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
    resume: bool = True,
) -> None:
    seed_everything(42)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "experiment_data.csv"
    csv_headers = [
        "round", "mAP50", "mAP50-95", "avg_loss",
        "bits_down_raw", "bits_down_compressed", "latency_down_sim",
        "bits_up_raw", "bits_up_compressed", "latency_up_sim",
        "total_round_time"
    ]
    for i in range(len(client_yaml_list)):
        csv_headers.append(f"client_{i}_loss")
    
    # 初始化 CSV（如果不存在或不需要恢复）
    ckpt_path = out_dir / "checkpoint_latest.pt"
    if not resume or not ckpt_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    if torch.cuda.is_available() and device != 'cpu':
        if device.isdigit():
            device = f"cuda:{device}"
    else:
        device = 'cpu'

    print(f"🚀 FLQ-YOLOv8 (FreezeBN) | Device: {device} | Bits: {bits}")

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

    # 初始化 global_sd 和 template_sd（从 warmup 后的模型）
    global_sd = {k: v.cpu().clone() for k, v in init_model.model.state_dict().items()}
    template_sd = copy.deepcopy(global_sd)
    
    del init_model
    gc.collect()
    if warmup_dir.exists(): shutil.rmtree(warmup_dir)

    # --- 2. Clients & Checkpoint 恢复 ---
    client_trainers = []
    for yaml_path in client_yaml_list:
        trainer = ManualClientTrainer(str(model_path_to_use), yaml_path, device, batch, imgsz)
        client_trainers.append(trainer)
    
    # Checkpoint 恢复逻辑
    start_round = 0
    best_map = 0.0
    if resume and ckpt_path.exists():
        global_sd, template_sd, start_round, best_map = load_checkpoint(
            ckpt_path, client_trainers, device
        )
        if csv_path.exists() and start_round > 0:
            import pandas as pd
            existing_df = pd.read_csv(csv_path)
            if len(existing_df) > start_round:
                print(f"   [Warning] CSV has {len(existing_df)} rows, but resuming from round {start_round}")

    server_helper = FLQCompressor(device)
    dil_sim = DILSimulator(min_bw=2.0, max_bw=10.0, min_loss=0.0, max_loss=0.2)
    
    # === Server Strategy: Global LR (No Momentum to avoid explosion) ===
    server_lr = 1.0 # FreezeBN is stable, can try 1.0 first. If unstable, reduce to 0.8.

    # --- 3. Federated Loop ---
    for r in range(start_round, rounds):
        print(f"\n========== Round {r} / {rounds - 1} ==========")
        t_start = time.time()
        
        # == 下行广播模拟 ==
        num_params = sum(p.numel() for p in global_sd.values())
        bits_down_raw = num_params * 32
        bits_down_compressed = bits_down_raw
        latency_down, _, _ = dil_sim.simulate_transmission(bits_down_compressed)
        print(f"   [Downlink] {bits_down_compressed/1e6:.2f} Mb, Latency: {latency_down:.3f}s")

        # == 本地训练 ==
        cur_lr = 0.01 * (0.98 ** r)
        client_updates_dense = []
        round_loss_list = []
        
        bits_up_raw_total = 0
        bits_up_compressed_total = 0
        max_latency_up = 0
        
        for i, trainer in enumerate(client_trainers):
            local_sd, loss_stats, meta = trainer.train_epoch(
                global_sd, local_epochs, lr=cur_lr)

            flat_global = server_helper.flatten_params(global_sd).to(device)
            flat_local = trainer.compressor.flatten_params(local_sd).to(device)
            delta = flat_local - flat_global

            q_delta, bit_cost = trainer.compressor.quantize_update(delta, bits)
            client_updates_dense.append(q_delta.cpu())
            
            client_loss = meta["final_loss"]
            round_loss_list.append(client_loss)
            
            raw_bits = delta.numel() * 32
            bits_up_raw_total += raw_bits
            bits_up_compressed_total += bit_cost
            
            lat, _, _ = dil_sim.simulate_transmission(bit_cost)
            max_latency_up = max(max_latency_up, lat)

            if i == 0:
                print(f"   [Client {i}] Loss: {client_loss:.4f} | Up: {bit_cost/1e6:.2f} Mb | Lat: {lat:.3f}s")
            
            # 及时清理中间变量
            del local_sd, flat_global, flat_local, delta, q_delta
            if (r + 1) % 10 == 0:
                trainer.compressor.reset_error()
            torch.cuda.empty_cache()

        avg_loss = np.mean(round_loss_list)

        # == 服务器聚合 ==
        print(f"   [Server] Aggregating (Global LR: {server_lr})...")
        client_updates_gpu = [u.to(device) for u in client_updates_dense]
        stack_updates = torch.stack(client_updates_gpu)
        avg_update = stack_updates.mean(dim=0)
        
        flat_global_new = server_helper.flatten_params(global_sd).to(device) + (server_lr * avg_update)
        global_sd = server_helper.reconstruct_state_dict(flat_global_new, template_sd)
        
        # 清理聚合相关的临时变量
        del client_updates_dense, client_updates_gpu, stack_updates, avg_update, flat_global_new
        torch.cuda.empty_cache()

        # == 评估 ==
        print("   [Server] Evaluating...")
        metrics = {"mAP50": 0, "mAP50-95": 0}
        try:
            torch.cuda.empty_cache()
            gc.collect()
            val_model = YOLO(str(model_path_to_use))
            val_model.model.load_state_dict(global_sd)
            results = val_model.val(
                data=str(val_yaml), batch=batch, device=device,
                verbose=False, plots=False
            )
            metrics["mAP50"] = results.results_dict.get("metrics/mAP50(B)", 0.0)
            metrics["mAP50-95"] = results.results_dict.get("metrics/mAP50-95(B)", 0.0)
            del results
            del val_model.model
            del val_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"   [Warning] Eval failed: {e}")
            import traceback
            traceback.print_exc()

        torch.cuda.empty_cache()
        
        t_end = time.time()
        round_time = t_end - t_start

        # == 记录与保存 ==
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

        if metrics["mAP50"] > best_map:
            best_map = metrics["mAP50"]
            torch.save(global_sd, out_dir / "global_best.pt")
            print(f"   [Save] New Best Model (mAP50={best_map:.4f})")

        print(f"   [Result] mAP50: {metrics['mAP50']:.4f} | Loss: {avg_loss:.4f} | Time: {round_time:.1f}s")
        
        # Save Checkpoint（每 5 个 round 保存一次，或最后一轮）
        if (r + 1) % 5 == 0 or r == rounds - 1:
            save_checkpoint(global_sd, template_sd, client_trainers, r, best_map, out_dir, device)
        
        # 每个 round 结束后都清理一次显存
        torch.cuda.empty_cache()
        gc.collect()

    torch.save(global_sd, out_dir / "global_last.pt")
    save_checkpoint(global_sd, template_sd, client_trainers, rounds - 1, best_map, out_dir, device)
    print(f"\n训练完成. 结果保存在: {out_dir}")

def generate_out_dir(model_path: str, bits: int, local_epochs: int, base_dir: str = "./results/runs_flq_SGD") -> str:
    """根据参数自动生成详细的输出文件夹名称"""
    # 从模型路径提取模型名称（如 yolov8s.pt -> yolov8s）
    model_name = Path(model_path).stem
    # 移除可能的路径前缀，只保留文件名
    if '/' in model_name or '\\' in model_name:
        model_name = Path(model_name).stem
    
    # 组合文件夹名称：runs_flq_SGD_yolov8s_32bit_1epoch
    detailed_name = f"runs_flq_SGD_{model_name}_{bits}bit_{local_epochs}epoch"
    
    # 如果 base_dir 是默认值，使用详细名称；否则使用用户指定的
    if base_dir == "./results/runs_flq_SGD":
        return f"./results/{detailed_name}"
    else:
        # 用户指定了自定义路径，直接使用
        return base_dir

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=str, nargs="+", required=True)
    p.add_argument("--val-data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="./results/runs_flq_SGD",
                   help="输出目录，默认会根据模型、压缩比、epoch自动生成详细名称")
    p.add_argument("--no-resume", action="store_true", help="不从 checkpoint 恢复，从头开始")
    return p.parse_args()

def main():
    args = parse_args()
    
    # 自动生成详细的输出文件夹名称
    out_dir = generate_out_dir(args.model, args.bits, args.local_epochs, args.out_dir)
    print(f"📁 Output directory: {out_dir}")
    
    run_federated_flq(
        [Path(p) for p in args.clients], Path(args.val_data), Path(args.model),
        args.rounds, args.local_epochs, args.bits, args.batch, args.imgsz,
        args.device, args.workers, Path(out_dir), resume=not args.no_resume
    )

if __name__ == "__main__":
    main()

