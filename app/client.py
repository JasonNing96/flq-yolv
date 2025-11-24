"""
FLQ-Fed 联邦学习客户端
核心逻辑移植自 flq_yolov_v8.py (FreezeBN Version)
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import requests
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict, Tuple
from types import SimpleNamespace

# Ultralytics 组件
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.data.utils import check_det_dataset
from torch.cuda.amp import autocast, GradScaler

from .config import Config
from .model_utils import (
    FLQCompressor,
    state_dict_to_vector,
    state_dict_to_grad_vector
)

# ==================== 全局配置 ====================

PROJECT_ROOT = Path(__file__).parent.parent
CLIENT_ID = None
SERVER_URL = None
DATA_YAML = None
OUTPUT_DIR = None


# ==================== 工具函数 ====================

def _ts():
    """时间戳"""
    return datetime.now().strftime('%H:%M:%S')


def _log(msg: str):
    """日志输出"""
    print(f"[{_ts()}] {msg}")


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 训练内核 (移植自 v8) ====================

class ManualClientTrainer:
    """
    手动训练循环，支持 FreezeBN
    移植自 flq_yolov_v8.py
    """
    def __init__(self, model_path: str, data_yaml: str, device: str, batch: int, imgsz: int):
        self.device = device
        self.data_yaml = data_yaml
        self.model_path = model_path
        self.batch = batch
        self.imgsz = imgsz
        
        # 加载配置
        self.cfg = get_cfg(DEFAULT_CFG)
        self.cfg.data = data_yaml
        self.cfg.imgsz = imgsz
        self.cfg.batch = batch
        
        # 检查数据集
        self.data_info = check_det_dataset(data_yaml)
        self.train_path = self.data_info['train']
        self.batch_size = batch
        
        # 压缩器
        self.compressor = FLQCompressor(device)

    def train_epoch(self, global_sd: Dict, local_epochs: int, lr: float) -> Tuple[Dict, dict, dict]:
        """执行本地训练"""
        # 临时加载模型以获取结构
        temp_wrapper = YOLO(self.model_path)
        model = temp_wrapper.model
        del temp_wrapper
        torch.cuda.empty_cache()

        if hasattr(model, 'args') and isinstance(model.args, dict):
            model.args = SimpleNamespace(**model.args)
        
        # 加载全局参数
        model.load_state_dict(global_sd)
        model.to(self.device)
        
        # ================== FREEZE BN LOGIC (v8 核心) ==================
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False # Critical: Stop tracking stats
                module.eval() # Freeze behavior
        # ===============================================================

        for param in model.parameters():
            param.requires_grad = True
        
        # 初始化 Loss
        loss_fn = v8DetectionLoss(model)
        if hasattr(loss_fn, 'hyp'):
            if isinstance(loss_fn.hyp, dict):
                loss_fn.hyp = SimpleNamespace(**loss_fn.hyp)
            # 默认超参
            if not hasattr(loss_fn.hyp, 'box'): loss_fn.hyp.box = 7.5
            if not hasattr(loss_fn.hyp, 'cls'): loss_fn.hyp.cls = 0.5
            if not hasattr(loss_fn.hyp, 'dfl'): loss_fn.hyp.dfl = 1.5

        if hasattr(loss_fn, 'proj'):
            loss_fn.proj = loss_fn.proj.to(self.device)

        # 优化器
        scaler = GradScaler()
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.0, weight_decay=5e-4)

        # 构建 DataLoader (每次重新构建以节省持久显存)
        dataset = build_yolo_dataset(
            self.cfg, self.train_path, self.batch_size, self.data_info, mode="train", rect=False, stride=32
        )
        loader = build_dataloader(
            dataset, self.batch_size, workers=0, shuffle=True, rank=-1)

        loss_stats = {"box": [], "cls": [], "dfl": []}

        # 训练循环
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
                
                # 显存清理
                del preds, loss, loss_items
                for k in list(batch.keys()):
                    if isinstance(batch[k], torch.Tensor):
                        del batch[k]
                del batch
                
                if (batch_idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()

        # 提取训练后的参数
        final_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 资源清理
        optimizer.zero_grad(set_to_none=True)
        del model, loss_fn, optimizer, scaler, loader, dataset
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        metadata = {
            "final_loss": np.mean(loss_stats["box"]) if loss_stats["box"] else 0.0,
            "metrics": {} # 手动训练暂不包含 eval metrics，由 server 统一 eval
        }
        return final_sd, loss_stats, metadata


# ==================== 核心流程 ====================

def pull_global_model(model_path: str) -> tuple:
    """
    从服务器拉取全局模型参数
    """
    try:
        response = requests.get(f"{SERVER_URL}/global", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # 反序列化 state_dict
        global_state_dict = {k: torch.tensor(v) for k, v in data['state_dict'].items()}
        
        current_round = data['round']
        is_done = data['done']
        
        _log(f"📥 拉取全局模型成功 (Round {current_round})")
        return current_round, is_done, global_state_dict
        
    except Exception as e:
        _log(f"❌ 拉取模型失败: {e}")
        raise


def push_update(
    trainer: ManualClientTrainer,
    local_sd: Dict,
    global_sd: Dict,
    n_samples: int,
    round_id: int,
    metadata: Dict,
    config: Config
):
    """
    压缩并上传更新
    """
    try:
        # 1. 计算差值 delta = local - global
        # 使用 compressor 的 helper
        flat_global = trainer.compressor.flatten_params(global_sd) # on device
        flat_local = trainer.compressor.flatten_params(local_sd)   # on device
        delta = flat_local - flat_global
        
        # 2. 量化
        bits = config.quant_bits if config.quant_enabled else 32
        q_delta, bit_cost, scale, zero_point = trainer.compressor.quantize_update(delta, bits)
        
        # 3. 序列化
        serialized_grad = q_delta.cpu().tolist()
        
        # 4. 构建 payload
        quant_params = {
            "scale": scale,
            "zero_point": zero_point,
            "bits": bits
        }
        
        payload = {
            "client_id": CLIENT_ID,
            "grad_vector": serialized_grad,
            "n_samples": n_samples,
            "round_id": round_id,
            "metrics": metadata.get("metrics", {}),
            "bits_up": bit_cost,
            "quant_params": quant_params
        }
        
        _log(f"📤 上传本地更新 (Bits Up: {bit_cost / 1e6:.2f} Mb)...")
        response = requests.post(f"{SERVER_URL}/update", json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        _log(f"✅ 上传成功 (Round {result['round']}, 缓冲={result['buffered']})")
        
        # 清理显存
        del flat_global, flat_local, delta, q_delta
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        _log(f"❌ 上传失败: {e}")
        raise


def count_samples(data_yaml_path: str) -> int:
    """统计本地训练样本数"""
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)
    
    # 统计训练集图片数
    train_path = Path(cfg['path']) / cfg['train']
    if train_path.exists():
        return len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png')))
    return 0


# ==================== 主流程 ====================

def start_client(client_id: int, server_url: str = None, config_path: Optional[str] = None):
    """
    启动联邦学习客户端
    """
    global CLIENT_ID, SERVER_URL, DATA_YAML, OUTPUT_DIR
    
    seed_everything(42)
    
    print("="*70)
    print(f"🚀 FLQ客户端 #{client_id} (v8: FreezeBN)")
    print("="*70)
    
    # 加载配置
    config = Config(config_path)
    
    # 设置全局变量
    CLIENT_ID = client_id
    SERVER_URL = server_url or f"http://{config.server_host}:{config.server_port}"
    DATA_YAML = str(PROJECT_ROOT / "data" / f"client{client_id}" / "oil.yaml")
    OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / f"client{client_id}" / "runs")
    
    _log(f"🌐 服务器: {SERVER_URL}")
    _log(f"📁 数据: {DATA_YAML}")
    _log(f"🖥️  设备: {config.device}")
    
    if not os.path.exists(DATA_YAML):
        _log(f"❌ 数据配置不存在: {DATA_YAML}")
        return
    
    n_samples = count_samples(DATA_YAML)
    _log(f"📊 本地样本数: {n_samples}\n")
    
    # 初始化 Trainer
    model_path = str(PROJECT_ROOT / config.model_name)
    trainer = ManualClientTrainer(
        model_path, DATA_YAML, config.device, config.batch_size, 640
    )
    
    last_round = -1
    
    while True:
        try:
            # 1. 拉取全局模型
            current_round, is_done, global_sd = pull_global_model(model_path)
            
            if is_done:
                print("\n" + "="*70)
                _log("🎉 训练完成！")
                print("="*70)
                break
            
            if current_round == last_round:
                time.sleep(5)
                continue
            
            _log(f"🎯 开始本地训练 Round {current_round} (LR={0.01 * (0.98 ** current_round):.5f})...")
            
            # 2. 本地训练 (v8 逻辑)
            cur_lr = 0.01 * (0.98 ** current_round)
            local_sd, _, metadata = trainer.train_epoch(
                global_sd, config.local_epochs, lr=cur_lr
            )
            
            # 3. 上传更新
            response = push_update(
                trainer, local_sd, global_sd, n_samples, current_round, metadata, config
            )
            
            last_round = current_round
            
            if response.get("done", False):
                break
                
            # 定期重置误差反馈 (v8 逻辑: 每10轮重置)
            if (current_round + 1) % 10 == 0:
                trainer.compressor.reset_error()
            
            _log(f"✅ Round {current_round} 完成\n")
        
        except KeyboardInterrupt:
            _log("⚠️  用户中断")
            break
        
        except Exception as e:
            _log(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m app.client <client_id>")
        sys.exit(1)
    
    start_client(int(sys.argv[1]))
