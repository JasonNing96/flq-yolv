"""
FLQ-Fed 联邦学习客户端
简化版 - 线性流程，易于调试
"""
import os
import time
import torch
import yaml
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict
from ultralytics import YOLO

from .config import Config
from .model_utils import (
    compute_model_size,
    quantize_vector, dequantize_vector,
    state_dict_to_vector, vector_to_state_dict,
    state_dict_to_grad_vector, grad_vector_to_state_dict,
    ErrorFeedback
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


# ==================== 核心流程 ====================

def pull_global_model(model: YOLO) -> tuple:
    """
    从服务器拉取全局模型
    
    Returns:
        current_round: 当前轮次
        is_done: 是否完成训练
        global_state_dict: 全局模型的 state_dict
    """
    try:
        response = requests.get(f"{SERVER_URL}/global", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 反序列化 state_dict
        global_state_dict = {k: torch.tensor(v) for k, v in data['state_dict'].items()}
        
        # 如果服务器下发的是量化模型，则需要反量化
        downlink_quant_bits = data.get('downlink_quant_bits', 0)
        if downlink_quant_bits > 0:
            _log(f"📥 服务器下发 {downlink_quant_bits}-bit 量化模型，进行反量化...")
            global_vector = state_dict_to_vector(global_state_dict)
            # 这里假设服务器已经反量化回全精度，客户端直接加载即可
            # 如果服务器下发的是量化值，这里需要 dequantize_vector
            # 但目前服务器端是先量化再反量化，所以客户端直接加载即可
        
        model.model.load_state_dict(global_state_dict, strict=False)
        
        current_round = data['round']
        is_done = data['done']
        
        _log(f"📥 拉取全局模型成功 (Round {current_round})")
        return current_round, is_done, global_state_dict
        
    except Exception as e:
        _log(f"❌ 拉取模型失败: {e}")
        raise


def train_local(model: YOLO, round_id: int, config: Config):
    """
    本地训练
    
    Args:
        model: YOLO模型
        round_id: 当前轮次
        config: 配置对象
    """
    _log(f"🎯 开始本地训练 Round {round_id}...")
    
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=config.local_epochs,
            batch=config.batch_size,
            imgsz=640,
            device=config.device,
            workers=config.workers,
            project=OUTPUT_DIR,
            name=f"round_{round_id}",
            exist_ok=True,
            verbose=config.verbose,
            val=config.enable_val,
            plots=config.enable_plots
        )
        
        # 提取关键指标
        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        map50 = metrics.get('metrics/mAP50(B)', 0.0)
        
        _log(f"✅ 本地训练完成 (mAP50: {map50:.3f})")
        return results
        
    except Exception as e:
        _log(f"❌ 训练失败: {e}")
        raise


def push_update(
    model: YOLO,
    n_samples: int,
    round_id: int,
    train_results: Any,
    last_global_state: Dict[str, Any],
    config: Config,
    error_feedback_instance: Optional[ErrorFeedback] = None
):
    """
    上传本地更新到服务器
    
    Args:
        model: 训练后的模型
        n_samples: 本地样本数
        round_id: 当前轮次
        train_results: 本地训练结果对象
        last_global_state: 上一轮的全局模型 state_dict
        config: 配置对象
        error_feedback_instance: 误差反馈实例
    """
    try:
        local_state_dict = model.model.state_dict()
        
        # 计算梯度差异
        grad_vector = state_dict_to_grad_vector(local_state_dict, last_global_state)
        
        bits_up = 0.0
        quant_params = None
        
        if config.aggregation_mode == "flq-fed" and config.quant_enabled:
            _log(f"🗜️  进行 {config.quant_bits}-bit 量化...")
            
            if error_feedback_instance and config.error_feedback_enabled:
                quantized_grad_vector, scale, zero_point = \
                    error_feedback_instance.compress_with_feedback(grad_vector, bits=config.quant_bits)
            else:
                quantized_grad_vector, scale, zero_point = \
                    quantize_vector(grad_vector, bits=config.quant_bits)
            
            # 序列化量化后的梯度向量
            serialized_grad_vector = quantized_grad_vector.cpu().tolist()
            
            # 计算上传比特数
            num_params = grad_vector.numel()
            bits_up = num_params * config.quant_bits
            
            quant_params = {
                "scale": scale,
                "zero_point": zero_point,
                "bits": config.quant_bits
            }
            
            _log(f"✅ 量化完成，上传 {config.quant_bits}-bit 梯度差异。")
        else:
            # FedAvg 或未启用量化，上传全精度梯度差异
            _log("⬆️  上传全精度梯度差异...")
            serialized_grad_vector = grad_vector.cpu().tolist()
            
            # 计算上传比特数 (32-bit 浮点数)
            num_params = grad_vector.numel()
            bits_up = num_params * 32
        
        # 提取训练指标
        metrics = {}
        if hasattr(train_results, 'results_dict'):
            results_dict = train_results.results_dict
            metrics['map50'] = results_dict.get('metrics/mAP50(B)', 0.0)
            metrics['map'] = results_dict.get('metrics/mAP50-95(B)', 0.0)
            metrics['precision'] = results_dict.get('metrics/precision(B)', 0.0)
            metrics['recall'] = results_dict.get('metrics/recall(B)', 0.0)
            metrics['loss'] = results_dict.get('train/box_loss', 0.0) + \
                              results_dict.get('train/cls_loss', 0.0) + \
                              results_dict.get('train/dfl_loss', 0.0)
        
        # 发送更新
        payload = {
            "client_id": CLIENT_ID,
            "grad_vector": serialized_grad_vector,
            "n_samples": n_samples,
            "round_id": round_id,
            "metrics": metrics,
            "bits_up": bits_up,
            "quant_params": quant_params
        }
        
        _log(f"📤 上传本地更新 (mAP50: {metrics.get('map50', 0.0):.3f}, Bits Up: {bits_up / (1024**2) / 8:.2f} MB)...")
        response = requests.post(f"{SERVER_URL}/update", json=payload, timeout=60)
        if not response.ok:
            _log(f"⚠️  上传失败详情: {response.text}")
        response.raise_for_status()
        
        result = response.json()
        _log(f"✅ 上传成功 (Round {result['round']}, 缓冲={result['buffered']})")
        
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
    
    Args:
        client_id: 客户端ID (1, 2, 3, ...)
        server_url: 服务器地址（可选）
        config_path: 配置文件路径（可选）
    """
    global CLIENT_ID, SERVER_URL, DATA_YAML, OUTPUT_DIR
    
    print("="*70)
    print(f"🚀 FLQ客户端 #{client_id}")
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
    _log(f"📂 输出: {OUTPUT_DIR}")
    _log(f"🖥️  设备: {config.device}")
    
    # 检查数据文件
    if not os.path.exists(DATA_YAML):
        _log(f"❌ 数据配置不存在: {DATA_YAML}")
        _log(f"💡 提示: 请先运行 python scripts/split_dataset.py")
        return
    
    # 统计样本数
    n_samples = count_samples(DATA_YAML)
    _log(f"📊 本地样本数: {n_samples}\n")
    
    # 初始化模型
    _log("📦 初始化模型...")
    model_path = PROJECT_ROOT / config.model_name
    model = YOLO(str(model_path))
    
    # 读取类别数
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg.get('nc', 80)
    
    from ultralytics.nn.tasks import DetectionModel
    model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)
    _log(f"✅ 模型初始化完成 (nc={nc})\n")
    
    # 主循环
    last_round = -1
    last_global_state = None
    error_feedback_instance = ErrorFeedback() if config.error_feedback_enabled else None
    
    while True:
        try:
            # 1. 拉取全局模型
            current_round, is_done, global_state_dict = pull_global_model(model)
            last_global_state = global_state_dict # 保存当前全局模型，用于计算梯度差异
            
            # 检查是否完成
            if is_done:
                print("\n" + "="*70)
                _log("🎉 所有联邦训练轮次已完成！")
                _log(f"📁 训练结果: {OUTPUT_DIR}")
                print("="*70)
                break
            
            # 检查是否已训练过当前轮次
            if current_round == last_round:
                _log("⏳ 等待服务器聚合...")
                time.sleep(5)
                continue
            
            # 2. 本地训练
            train_results = train_local(model, current_round, config)
            
            # 3. 上传更新
            response = push_update(model, n_samples, current_round, train_results, last_global_state, config, error_feedback_instance)
            
            # 更新轮次记录
            last_round = current_round
            
            # 检查服务器返回的完成标志
            if response.get("done", False):
                print("\n" + "="*70)
                _log("🎉 联邦训练完成（服务器通知）")
                print("="*70)
                break
            
            _log(f"✅ Round {current_round} 完成\n")
        
        except KeyboardInterrupt:
            _log("⚠️  用户中断")
            break
        
        except Exception as e:
            _log(f"❌ 错误: {e}")
            _log("🔄 5秒后重试...")
            time.sleep(5)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m app.client <client_id>")
        sys.exit(1)
    
    start_client(int(sys.argv[1]))

