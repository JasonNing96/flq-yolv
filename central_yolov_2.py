#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manual Centralized YOLOv8 Training Script v2 (Aligned with FL)
----------------------------------------------------------------
- 目的: 作为 FL 实验的 "Upper Bound" 对照组，配置与 FL 对齐
- 改进:
  1. 添加 Warmup（与 FL 一致）
  2. 使用指数衰减学习率：0.01 * (0.98 ** epoch)
  3. 每个 epoch 都验证（而非每 5 个）
  4. 支持独立的验证集参数
  5. 保持 checkpoint 和 CSV 记录功能
----------------------------------------------------------------
"""

import argparse
import copy
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
    def __init__(self, model_path: str, data_yaml: Path, val_yaml: Path, device: str, batch: int, imgsz: int):
        self.device = device
        self.data_yaml = data_yaml
        self.val_yaml = val_yaml
        self.model_path = model_path
        self.batch = batch
        self.imgsz = imgsz

        # Load Training Data
        cfg = get_cfg(DEFAULT_CFG)
        cfg.data = str(data_yaml)
        cfg.imgsz = imgsz
        cfg.batch = batch
        data_info = check_det_dataset(str(data_yaml))
        train_path = data_info['train']
        
        print(f"[Data] Loading centralized training data from: {train_path}")
        self.dataset = build_yolo_dataset(
            cfg, train_path, batch, data_info, mode="train", rect=False, stride=32
        )
        self.loader = build_dataloader(
            self.dataset, batch, workers=4, shuffle=True, rank=-1) # 集中式可以用多workers

    def save_checkpoint(self, model, optimizer, scaler, epoch, best_map, out_dir: Path):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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

    def load_checkpoint(self, checkpoint_path: Path, model, optimizer, scaler):
        """加载 checkpoint"""
        if not checkpoint_path.exists():
            return None, 0, 0.0
        
        print(f"[Checkpoint] Loading from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        best_map = checkpoint.get('best_map', 0.0)
        
        print(f"[Checkpoint] Resumed from epoch {checkpoint['epoch']}, best_map={best_map:.4f}")
        return start_epoch, best_map

    def run_training(self, epochs: int, lr0: float, out_dir: Path, resume: bool = True, use_warmup: bool = True):
        # === 1. Warmup (与 FL 对齐) ===
        model_path_to_use = self.model_path
        if use_warmup:
            print("   [Init] Adapting model head (Warmup)...")
            warmup_dir = out_dir / "warmup_temp"
            init_model = YOLO(str(self.model_path))
            try:
                init_model.train(
                    data=str(self.val_yaml), epochs=1, imgsz=self.imgsz, batch=self.batch,
                    device=self.device, project=str(warmup_dir), name="init_run",
                    exist_ok=True, plots=False, save=True, val=False, verbose=False
                )
            except Exception as e:
                print(f"   [Init] Warmup passed: {e}")
            
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ.pop("CUDA_VISIBLE_DEVICES")

            warmup_pt = warmup_dir / "init_run/weights/last.pt"
            adapted_pt = out_dir / "init_adapted.pt"
            if warmup_pt.exists():
                shutil.copy(str(warmup_pt), str(adapted_pt))
                model_path_to_use = adapted_pt
                del init_model
                print(f"   [Init] Warmup completed, using adapted model: {adapted_pt}")
            else:
                print("   [Warning] Warmup failed, using original model.")
                del init_model
            
            if warmup_dir.exists():
                shutil.rmtree(warmup_dir)
            gc.collect()
            torch.cuda.empty_cache()

        # === 2. Model Init ===
        temp_wrapper = YOLO(str(model_path_to_use))
        model = temp_wrapper.model
        del temp_wrapper
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

        # Optimizer (与 FL 对齐：SGD with momentum)
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.937, weight_decay=5e-4)
        scaler = GradScaler()
        
        # 注意：不使用 scheduler，而是手动计算指数衰减学习率（与 FL 对齐）

        # === 3. Checkpoint 恢复 ===
        start_epoch = 0
        best_map = 0.0
        
        csv_path = out_dir / "central_log.csv"
        
        if resume:
            ckpt_path = out_dir / "checkpoint_latest.pt"
            if ckpt_path.exists():
                start_epoch, best_map = self.load_checkpoint(ckpt_path, model, optimizer, scaler)
                
                # 如果 CSV 已存在，检查是否需要追加
                if csv_path.exists() and start_epoch > 0:
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
        print(f"[Train] Using exponential LR decay: lr = {lr0} * (0.98 ^ epoch)")
        
        try:
            for epoch in range(start_epoch, epochs):
                t_start = time.time()
                model.train()
                
                # === 学习率指数衰减（与 FL 对齐）===
                current_lr = lr0 * (0.98 ** epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
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
                    
                    loss_sum += loss_items[0].item()
                    loss_count += 1
                    
                    # 及时释放显存
                    del preds, loss, loss_items
                    for k in list(batch.keys()):
                        if isinstance(batch[k], torch.Tensor):
                            del batch[k]
                    del batch
                    
                    # 每 100 个 batch 清理一次显存
                    if (batch_idx + 1) % 100 == 0:
                        torch.cuda.empty_cache()
                
                avg_loss = loss_sum / loss_count if loss_count > 0 else 0.0
                
                # === 验证（每个 epoch 都验证，与 FL 对齐）===
                mAP50 = 0.0
                mAP95 = 0.0
                print(f"[Validate] Running validation at epoch {epoch+1}...")
                try:
                    model.eval()
                    torch.cuda.empty_cache()
                    
                    # 创建验证 wrapper
                    val_wrapper = YOLO(str(model_path_to_use))
                    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    val_wrapper.model.load_state_dict(state_dict)
                    
                    # 使用独立的验证集
                    results = val_wrapper.val(
                        data=str(self.val_yaml), batch=self.batch, device=self.device,
                        verbose=False, plots=False
                    )
                    mAP50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
                    mAP95 = results.results_dict.get("metrics/mAP50-95(B)", 0.0)
                    
                    # 清理验证结果
                    del results
                    del state_dict
                    del val_wrapper
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    model.train()
                except Exception as e:
                    print(f"   [Warning] Validation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    model.train()

                t_end = time.time()
                duration = t_end - t_start
                
                print(f"Epoch {epoch+1}/{epochs}: mAP50={mAP50:.4f} | mAP50-95={mAP95:.4f} | Loss={avg_loss:.4f} | LR={current_lr:.6f} | Time={duration:.1f}s")
                
                # Save CSV (追加模式)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, mAP50, mAP95, avg_loss, current_lr, duration])
                
                if mAP50 > best_map:
                    best_map = mAP50
                    torch.save(model.state_dict(), out_dir / "best_central.pt")
                    print(f"   [Save] New Best Model (mAP50={best_map:.4f})")
                
                # 每 10 个 epoch 保存一次 checkpoint
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    self.save_checkpoint(model, optimizer, scaler, epoch, best_map, out_dir)
                
                # 每个 epoch 结束后都清理一次显存
                torch.cuda.empty_cache()
                gc.collect()
                    
        except KeyboardInterrupt:
            print("\n[Interrupted] Saving checkpoint before exit...")
            self.save_checkpoint(model, optimizer, scaler, epoch, best_map, out_dir)
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"\n[Error] Training failed at epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            print("[Error] Saving checkpoint before exit...")
            self.save_checkpoint(model, optimizer, scaler, epoch, best_map, out_dir)
            torch.cuda.empty_cache()
            raise
        finally:
            torch.cuda.empty_cache()
                
        torch.save(model.state_dict(), out_dir / "last_central.pt")
        print(f"\nDone. Best mAP50: {best_map:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Centralized YOLOv8 Training v2 (Aligned with FL)")
    parser.add_argument("--data", type=str, required=True,
                        help="Training data YAML file (e.g., ./data/oil_detection_dataset/data.yaml)")
    parser.add_argument("--val-data", type=str, required=True,
                        help="Validation data YAML file (e.g., ./data/oil_detection_dataset/data.yaml)")
    parser.add_argument("--model", type=str, default="./models/yolov8s.pt",
                        help="Model path (default: ./models/yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--out-dir", type=str, default="./results/central_manual_2",
                        help="Output directory (default: ./results/central_manual_2)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (default: cuda:0)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume from checkpoint, start from scratch")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Disable warmup (default: enabled)")
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    
    trainer = CentralizedTrainer(
        args.model, Path(args.data), Path(args.val_data), 
        args.device, args.batch, imgsz=640
    )
    trainer.run_training(
        args.epochs, lr0=0.01, out_dir=Path(args.out_dir), 
        resume=not args.no_resume, use_warmup=not args.no_warmup
    )

if __name__ == "__main__":
    main()
