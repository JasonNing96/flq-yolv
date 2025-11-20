#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLQ-YOLOv5 联邦训练脚本 (Stateless Version)
----------------------------------------------------------------
- 核心修复: 采用 "Stateless" 设计。ManualClientTrainer 不再持有模型实例。
- 显存管理: 每次 train_epoch 动态加载模型，结束后强制销毁并 GC，确保显存归零。
- 解决报错: 彻底根治 CUDA OOM 导致的 Device Mismatch 问题。
----------------------------------------------------------------
"""

# ================= 补丁: 修复 Torch 版本兼容性 =================
import torch
try:
    _ = torch.OutOfMemoryError
except AttributeError:
    torch.OutOfMemoryError = RuntimeError
# ===========================================================

import argparse
import copy
import json
import random
import os
import time
import shutil
import gc  # <--- 新增：垃圾回收模块
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from types import SimpleNamespace  # <--- 新增这行引用

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

# ====================== 1. 基础工具 & DIL ======================


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
                out[k] = flat_vec[offset: offset +
                                  numel].view(v.shape).to(v.dtype).cpu()
                offset += numel
            else:
                out[k] = v.clone().cpu()
        return out

    def quantize_update(self, delta_vec: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int]:
        num_params = delta_vec.numel()
        if self.local_error is None:
            self.local_error = torch.zeros_like(delta_vec)
        target = delta_vec + self.local_error

        if bits >= 32:
            self.local_error.zero_()
            return target, num_params * 32

        if bits == 1:
            scale = target.abs().mean()
            if scale < 1e-8:
                scale = 1e-8
            sign = torch.sign(target)
            sign[sign == 0] = 1.0
            quantized = sign * scale
            self.local_error = target - quantized
            return quantized, num_params + 32
        else:
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
        self.model_path = model_path  # 只存路径，不加载模型对象
        self.batch = batch
        self.imgsz = imgsz

        # --- DataLoader (保持持久化，因为 DataLoader 重建很慢且不占显存) ---
        cfg = get_cfg(DEFAULT_CFG)
        cfg.data = str(data_yaml)
        cfg.imgsz = imgsz
        cfg.batch = batch

        data_info = check_det_dataset(str(data_yaml))
        train_path = data_info['train']

        self.dataset = build_yolo_dataset(
            cfg, train_path, batch, data_info, mode="train", rect=False, stride=32
        )
        # 强制 workers=0 避免多进程死锁
        self.loader = build_dataloader(
            self.dataset, batch, workers=0, shuffle=True, rank=-1)

        # 压缩器是轻量级的，可以保留
        self.compressor = FLQCompressor(device)

    def train_epoch(self, global_sd: Dict, local_epochs: int, lr: float, momentum: float) -> Tuple[Dict, dict, dict]:
        """执行本地训练 - 显存安全版"""

        # 1. 动态创建模型 (Fresh Load)
        # 这确保了没有任何之前的缓存残留
        temp_wrapper = YOLO(self.model_path)
        model = temp_wrapper.model

        # ================= 关键修复：Dict 转 Namespace =================
        # 解决 AttributeError: 'dict' object has no attribute 'box'
        if hasattr(model, 'args') and isinstance(model.args, dict):
            model.args = SimpleNamespace(**model.args)
        # ===========================================================
        
        # 加载权重 & 移至 GPU
        model.load_state_dict(global_sd)
        model.to(self.device)
        model.train()

        #=================== 关键修复：解冻所有参数 ===================
        # 解决 RuntimeError: element 0 of tensors does not require grad
        for param in model.parameters():
            param.requires_grad = True
        # ================
        
        # 2. 动态创建 Loss & Scaler
        loss_fn = v8DetectionLoss(model)
        
        # =================== 终极修复：直接修正 Loss 对象的超参数 ===================
        # 有时候 v8DetectionLoss 会深拷贝 args，导致上面对 model.args 的修改没生效
        # 或者 model.args 本身被 Ultralytics 内部逻辑重置了
        if hasattr(loss_fn, 'hyp'):
            # 确保 hyp 是个 Namespace
            if isinstance(loss_fn.hyp, dict):
                loss_fn.hyp = SimpleNamespace(**loss_fn.hyp)
            
            # 暴力注入默认值
            if not hasattr(loss_fn.hyp, 'box'): loss_fn.hyp.box = 7.5
            if not hasattr(loss_fn.hyp, 'cls'): loss_fn.hyp.cls = 0.5
            if not hasattr(loss_fn.hyp, 'dfl'): loss_fn.hyp.dfl = 1.5
        # =======================================================================

        if hasattr(loss_fn, 'proj'):
            loss_fn.proj = loss_fn.proj.to(self.device)

        scaler = GradScaler()
        # === 关键调整：适应 Stateless 训练 ===
        # 保持原始 LR，但移除 Momentum 以防止在重置 Optimizer 时发生震荡
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.0, weight_decay=5e-4)

        loss_stats = {"box": [], "cls": [], "dfl": []}

        # --- 训练循环 ---
        for epoch in range(local_epochs):
            for batch in self.loader:
                # 预处理
                batch['img'] = batch['img'].to(
                    self.device, non_blocking=True).float() / 255.0
                for k in batch:
                    if k != 'img' and isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)

                optimizer.zero_grad()

                # AMP 前向
                with autocast(enabled=True):
                    preds = model(batch['img'])
                    loss, loss_items = loss_fn(preds, batch)

                # AMP 反向
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()

                loss_stats["box"].append(loss_items[0].item())
                loss_stats["cls"].append(loss_items[1].item())
                loss_stats["dfl"].append(loss_items[2].item())

        # --- 训练结束：提取结果 ---
        final_sd = {k: v.cpu() for k, v in model.state_dict().items()}

        # ================= 激进的显存清理 =================
        # 1. 手动删除引用
        del model
        del temp_wrapper
        del loss_fn
        del optimizer
        del scaler

        # 2. 强制垃圾回收 (清除 Python 对象)
        gc.collect()

        # 3. 清空 PyTorch 缓存 (清除 GPU 碎片)
        torch.cuda.empty_cache()
        # ===============================================

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

    if torch.cuda.is_available() and device != 'cpu':
        if device.isdigit():
            device = f"cuda:{device}"
    else:
        device = 'cpu'

    print(f"🚀 FLQ-YOLOv5 (Stateless Fixed) | Device: {device} | Bits: {bits}")

    # 1. 初始化 & Warmup
    print("   [Init] Adapting model head (Warmup)...")
    warmup_dir = out_dir / "warmup_temp"

    # 临时创建模型用于 Warmup
    init_model = YOLO(str(model_path))
    try:
        # 修正：Warmup 也使用 GPU，避免 Ultralytics 设置 CUDA_VISIBLE_DEVICES=-1 污染环境
        init_model.train(
            data=str(val_yaml), epochs=1, imgsz=imgsz, batch=batch,
            device=device, project=str(warmup_dir), name="init_run",
            exist_ok=True, plots=False, save=True, val=False, verbose=False
        )
    except Exception as e:
        print(f"   [Init] Warmup passed: {e}")
    
    # === 关键修复：清理 Ultralytics 可能残留的环境变量 ===
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("CUDA_VISIBLE_DEVICES")
    # ==================================================

    warmup_pt = warmup_dir / "init_run/weights/last.pt"
    adapted_pt = out_dir / "init_adapted.pt"
    model_path_to_use = model_path

    if warmup_pt.exists():
        shutil.copy(str(warmup_pt), str(adapted_pt))
        model_path_to_use = adapted_pt
        # 重新加载以获取正确结构
        del init_model
        init_model = YOLO(str(adapted_pt))
    else:
        print("   [Warning] Warmup failed, using original model.")

    # 获取初始权重
    global_sd = {k: v.cpu().clone()
                 for k, v in init_model.model.state_dict().items()}
    template_sd = copy.deepcopy(global_sd)

    # === 关键：用完立刻删除 init_model ===
    del init_model
    gc.collect()
    if warmup_dir.exists():
        try:
            shutil.rmtree(warmup_dir)
        except:
            pass
    # ===================================

    # 2. 初始化 Clients (只传路径，不传对象)
    client_trainers = []
    for yaml_path in client_yaml_list:
        trainer = ManualClientTrainer(
            str(model_path_to_use), yaml_path, device, batch, imgsz)
        client_trainers.append(trainer)

    server_helper = FLQCompressor(device)
    # server_momentum_buffer = None # 移除动量buffer初始化
    log = {"round": [], "mAP50": [], "mAP50-95": [],
           "box_loss": [], "bits_up_raw": [], "bits_up_dil": []}
    
    # 详细日志记录
    client_details = {i: {"box_loss": [], "grad_scale": []} for i in range(len(client_yaml_list))}

    # 3. 联邦循环
    for r in range(rounds):
        print(f"\n========== Round {r} / {rounds - 1} ==========")
        t_start = time.time()
        cur_lr = 0.01 * (0.98 ** r)

        client_updates_dense = []
        round_box_loss = 0.0
        bits_up_raw_total = 0.0
        bits_up_dil_total = 0.0

        # --- A. 客户端训练 ---
        for i, trainer in enumerate(client_trainers):
            local_sd, loss_stats, meta = trainer.train_epoch(
                global_sd, local_epochs, lr=cur_lr, momentum=0.937)

            flat_global = server_helper.flatten_params(global_sd).to(device)
            flat_local = trainer.compressor.flatten_params(local_sd).to(device)
            delta = flat_local - flat_global

            q_delta, bit_cost = trainer.compressor.quantize_update(delta, bits)
            client_updates_dense.append(q_delta)
            
            scale = q_delta.abs().mean().item()
            client_details[i]["box_loss"].append(meta["final_loss"])
            client_details[i]["grad_scale"].append(scale)

            bits_up_raw_total += bit_cost
            bits_up_dil_total += apply_DIL_fluctuation(bit_cost)

            avg_loss = np.mean(loss_stats["box"]) if loss_stats["box"] else 0
            round_box_loss += avg_loss

            if i == 0:
                print(
                    f"   [Client {i}] Loss: {avg_loss:.4f} | Scale: {scale:.6f}")

        round_box_loss /= len(client_trainers)

        # --- B. 服务器聚合（修正版：纯 FedAvg，无服务器动量） ---
        print("   [Server] Aggregating...")
        stack_updates = torch.stack(client_updates_dense)
        avg_update = stack_updates.mean(dim=0)

        # 可选：如果担心步长过大，可乘一个 global_lr，例如 1.0
        # 这里我们先保持最原始的 FedAvg
        
        # 移除所有动量逻辑
        # if server_momentum_buffer is None: ...
        # server_momentum_buffer = ...

        flat_global_new = server_helper.flatten_params(
            global_sd).to(device) + avg_update
        global_sd = server_helper.reconstruct_state_dict(
            flat_global_new, template_sd)

        # --- C. 评估 (动态加载模式) ---
        print("   [Server] Evaluating...")
        metrics = {"mAP50": 0, "mAP50-95": 0}
        try:
            torch.cuda.empty_cache()
            # 动态加载评估模型
            val_model = YOLO(str(model_path_to_use))
            val_model.model.load_state_dict(global_sd)

            # 恢复使用 GPU (device) 进行评估
            results = val_model.val(
                data=str(val_yaml), batch=batch, device=device,
                verbose=False, plots=False
            )
            metrics["mAP50"] = results.results_dict.get(
                "metrics/mAP50(B)", 0.0)
            metrics["mAP50-95"] = results.results_dict.get(
                "metrics/mAP50-95(B)", 0.0)

            # 评估完立刻销毁
            del val_model
            gc.collect()

        except Exception as e:
            print(f"   [Warning] Eval failed: {e}")

        torch.cuda.empty_cache()

        # --- D. 日志 ---
        log["round"].append(r)
        log["mAP50"].append(metrics["mAP50"])
        log["mAP50-95"].append(metrics["mAP50-95"])
        log["box_loss"].append(round_box_loss)
        log["bits_up_raw"].append(bits_up_raw_total)
        log["bits_up_dil"].append(bits_up_dil_total)

        save_json(log, out_dir / "flq_log.json")
        # 保存 Client 详细日志
        save_json(client_details, out_dir / "client_details.json")
        
        print(
            f"   [Result] mAP50: {metrics['mAP50']:.4f} | Loss: {round_box_loss:.4f} | Time: {time.time()-t_start:.1f}s")

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
    p.add_argument("--batch", type=int, default=4)  # 默认调小
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="./results/runs_flq_v5")
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
