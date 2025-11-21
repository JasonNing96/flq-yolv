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

    def run_training(self, epochs: int, lr0: float, out_dir: Path):
        # 1. Model Init
        temp_wrapper = YOLO(self.model_path)
        model = temp_wrapper.model
        
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
        # Centralized 既然是连续训练，可以使用 Momentum!
        # 这是最大的区别：FL 因为是 stateless 所以把 momentum 关了，
        # 但 Centralized 可以保留 optimizer state。
        # 为了公平对比 FL (v7 已加 server momentum)，这里我们开启 standard SGD momentum
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.937, weight_decay=5e-4)
        scaler = GradScaler()
        
        # Scheduler (Linear/Cosine decay)
        # 简单起见，模拟 FL 的 step decay
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

        best_map = 0.0
        csv_path = out_dir / "central_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "mAP50", "mAP50-95", "loss", "lr", "time"])

        print(f"[Train] Starting centralized training for {epochs} epochs...")
        
        for epoch in range(epochs):
            t_start = time.time()
            model.train()
            loss_list = []
            
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
                
                loss_list.append(loss_items[0].item()) # box loss as proxy
            
            scheduler.step()
            avg_loss = np.mean(loss_list)
            
            # Validate
            mAP50 = 0.0
            mAP95 = 0.0
            if (epoch + 1) % 1 == 0: # Every epoch val
                try:
                    val_model = YOLO(self.model_path) # Re-load structure
                    # Load current weights
                    val_model.model.load_state_dict(model.state_dict())
                    results = val_model.val(
                        data=str(self.data_yaml), batch=self.batch, device=self.device,
                        verbose=False, plots=False
                    )
                    mAP50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
                    mAP95 = results.results_dict.get("metrics/mAP50-95(B)", 0.0)
                    del val_model
                except Exception as e:
                    print(f"Val failed: {e}")

            t_end = time.time()
            duration = t_end - t_start
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: mAP50={mAP50:.4f} | Loss={avg_loss:.4f} | LR={current_lr:.5f} | Time={duration:.1f}s")
            
            # Save
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, mAP50, mAP95, avg_loss, current_lr, duration])
            
            if mAP50 > best_map:
                best_map = mAP50
                torch.save(model.state_dict(), out_dir / "best_central.pt")
                
        torch.save(model.state_dict(), out_dir / "last_central.pt")
        print(f"Done. Best mAP50: {best_map:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/oil_detection_dataset/data.yaml")
    parser.add_argument("--model", type=str, default="./models/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="./results/central_manual")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    
    trainer = CentralizedTrainer(
        args.model, Path(args.data), args.device, args.batch, imgsz=640
    )
    trainer.run_training(args.epochs, lr0=0.01, out_dir=Path(args.out_dir))

if __name__ == "__main__":
    main()

