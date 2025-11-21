#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manual Centralized YOLOv8 Training Script (Benchmark)
----------------------------------------------------------------
- 目的: 作为 FL 实验的 "Upper Bound" 对照组
- 特点: 
  1. 使用与 FL 相同的 ManualTrainer 代码结构 (Stateless, Custom SGD)
  2. 集中式训练 (Centralized): 所有数据都在一个 Client 上
  3. 排除 FL 因素 (通信、聚合、Drift)，只测试训练器本身的性能上限
  4. 支持断点续训 (Checkpoint/Resume)
----------------------------------------------------------------
"""

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
from types import SimpleNamespace
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Ultralytics
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.data.utils import check_det_dataset
import pandas as pd

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CentralizedTrainer:
    def __init__(self, model_path: str, data_yaml: Path, device: str, batch: int, imgsz: int):
        self.device = device
        self.data_yaml = data_yaml
        self.model_path = model_path
        self.batch = batch
        self.imgsz = imgsz

        # Load Data
        cfg = get_cfg(DEFAULT_CFG)
        cfg.data = str(data_yaml)
        cfg.imgsz = imgsz
        cfg.batch = batch
        data_info = check_det_dataset(str(data_yaml))
        train_path = data_info['train']
        
        print(f"[Data] Loading centralized data from: {train_path}")
        self.dataset = build_yolo_dataset(
            cfg, train_path, batch, data_info, mode="train", rect=False, stride=32
        )
        self.loader = build_dataloader(
            self.dataset, batch, workers=4, shuffle=True, rank=-1) # 集中式可以用多workers

    def save_checkpoint(self, model, optimizer, scheduler, scaler, epoch, best_map, out_dir: Path):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_map': best_map,
        }
        ckpt_path = out_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"[Checkpoint] Saved to {ckpt_path}")
        
        # 每 10 个 epoch 保存一次历史 checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path_hist = out_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, ckpt_path_hist)
            print(f"[Checkpoint] Saved history checkpoint to {ckpt_path_hist}")

    def load_checkpoint(self, checkpoint_path: Path, model, optimizer, scheduler, scaler):
        """加载 checkpoint"""
        if not checkpoint_path.exists():
            return None, 0, 0.0
        
        print(f"[Checkpoint] Loading from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        best_map = checkpoint.get('best_map', 0.0)
        
        print(f"[Checkpoint] Resumed from epoch {checkpoint['epoch']}, best_map={best_map:.4f}")
        return start_epoch, best_map

    def run_training(self, epochs: int, lr0: float, out_dir: Path, resume: bool = True):
        # 1. Model Init
        temp_wrapper = YOLO(self.model_path)
        model = temp_wrapper.model
        del temp_wrapper  # 立即释放 wrapper，只保留 model
        torch.cuda.empty_cache()
        
        # Namespace Fix
        if hasattr(model, 'args') and isinstance(model.args, dict):
            model.args = SimpleNamespace(**model.args)
            
        model.to(self.device)
        model.train()
        
        for param in model.parameters():
            param.requires_grad = True
            
        # Loss Init
        loss_fn = v8DetectionLoss(model)
        if hasattr(loss_fn, 'hyp'):
            if isinstance(loss_fn.hyp, dict):
                loss_fn.hyp = SimpleNamespace(**loss_fn.hyp)
            if not hasattr(loss_fn.hyp, 'box'): loss_fn.hyp.box = 7.5
            if not hasattr(loss_fn.hyp, 'cls'): loss_fn.hyp.cls = 0.5
            if not hasattr(loss_fn.hyp, 'dfl'): loss_fn.hyp.dfl = 1.5
        if hasattr(loss_fn, 'proj'):
            loss_fn.proj = loss_fn.proj.to(self.device)

        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.937, weight_decay=5e-4)
        scaler = GradScaler()
        
        # Scheduler: 使用 CosineAnnealingLR 或更合理的衰减策略
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr0 * 0.01)

        # 尝试恢复 checkpoint
        start_epoch = 0
        best_map = 0.0
        csv_path = out_dir / "central_log.csv"
        
        if resume:
            ckpt_path = out_dir / "checkpoint_latest.pt"
            if ckpt_path.exists():
                start_epoch, best_map = self.load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler)
                # 如果 CSV 已存在，检查是否需要追加
                if csv_path.exists() and start_epoch > 0:
                    # 读取已有数据，确保不重复
                    existing_df = pd.read_csv(csv_path)
                    if len(existing_df) > start_epoch:
                        print(f"[Warning] CSV has {len(existing_df)} rows, but resuming from epoch {start_epoch}")
            else:
                # 首次运行，创建新的 CSV
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "mAP50", "mAP50-95", "loss", "lr", "time"])
        else:
            # 不恢复，从头开始
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "mAP50", "mAP50-95", "loss", "lr", "time"])

        print(f"[Train] Starting centralized training for {epochs} epochs (from epoch {start_epoch})...")
        
        # 预先创建验证用的 YOLO wrapper（避免每个epoch都创建）
        val_wrapper = None
        
        try:
            for epoch in range(start_epoch, epochs):
                t_start = time.time()
                model.train()
                loss_sum = 0.0
                loss_count = 0
                
                for batch_idx, batch in enumerate(self.loader):
                    # Preprocess
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
                    
                    # 使用累加而不是列表，节省内存
                    loss_sum += loss_items[0].item()
                    loss_count += 1
                    
                    # 及时释放 batch 和 preds 的显存引用
                    del preds, loss, loss_items
                    for k in list(batch.keys()):
                        if isinstance(batch[k], torch.Tensor):
                            del batch[k]
                    del batch
                    
                    # 每 100 个 batch 清理一次显存（避免累积）
                    if (batch_idx + 1) % 100 == 0:
                        torch.cuda.empty_cache()
                
                scheduler.step()
                avg_loss = loss_sum / loss_count if loss_count > 0 else 0.0
                
                # Validate: 每 5 个 epoch 验证一次（减少验证频率，节省显存和时间）
                mAP50 = 0.0
                mAP95 = 0.0
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    try:
                        model.eval()  # 切换到评估模式
                        torch.cuda.empty_cache()  # 验证前清理
                        
                        # 只在第一次验证时创建 wrapper，后续复用
                        if val_wrapper is None:
                            val_wrapper = YOLO(self.model_path)
                        
                        # 加载当前权重
                        val_wrapper.model.load_state_dict(model.state_dict())
                        results = val_wrapper.val(
                            data=str(self.data_yaml), batch=self.batch, device=self.device,
                            verbose=False, plots=False
                        )
                        mAP50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
                        mAP95 = results.results_dict.get("metrics/mAP50-95(B)", 0.0)
                        
                        # 清理验证结果
                        del results
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        model.train()  # 切换回训练模式
                    except Exception as e:
                        print(f"Val failed: {e}")
                        import traceback
                        traceback.print_exc()
                        model.train()  # 确保回到训练模式

                t_end = time.time()
                duration = t_end - t_start
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs}: mAP50={mAP50:.4f} | Loss={avg_loss:.4f} | LR={current_lr:.6f} | Time={duration:.1f}s")
                
                # Save CSV (追加模式)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, mAP50, mAP95, avg_loss, current_lr, duration])
                
                if mAP50 > best_map:
                    best_map = mAP50
                    torch.save(model.state_dict(), out_dir / "best_central.pt")
                
                # 每 5 个 epoch 保存一次 checkpoint
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    self.save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_map, out_dir)
                
                # 每个 epoch 结束后都清理一次显存
                torch.cuda.empty_cache()
                gc.collect()
                    
        except KeyboardInterrupt:
            print("\n[Interrupted] Saving checkpoint before exit...")
            self.save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_map, out_dir)
            if val_wrapper is not None:
                del val_wrapper
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"\n[Error] Training failed at epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            print("[Error] Saving checkpoint before exit...")
            self.save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_map, out_dir)
            if val_wrapper is not None:
                del val_wrapper
            torch.cuda.empty_cache()
            raise
        finally:
            # 确保清理验证 wrapper
            if val_wrapper is not None:
                del val_wrapper
                torch.cuda.empty_cache()
                
        torch.save(model.state_dict(), out_dir / "last_central.pt")
        print(f"Done. Best mAP50: {best_map:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/oil_detection_dataset/data.yaml")
    parser.add_argument("--model", type=str, default="./models/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=200)  # 改为 200
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="./results/central_manual")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-resume", action="store_true", help="不从 checkpoint 恢复，从头开始")
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    
    trainer = CentralizedTrainer(
        args.model, Path(args.data), args.device, args.batch, imgsz=640
    )
    trainer.run_training(args.epochs, lr0=0.01, out_dir=Path(args.out_dir), resume=not args.no_resume)

if __name__ == "__main__":
    main()

