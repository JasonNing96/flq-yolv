"""
FLQ-Fed 联邦学习服务器
简化版 - 集中所有服务器逻辑
"""
import os
import torch
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

from .config import Config
from .model_utils import (
    state_dict_to_vector, vector_to_state_dict,
    fedavg_aggregate, compute_model_size, compute_compression_ratio,
    state_dict_to_grad_vector, grad_vector_to_state_dict,
    quantize_vector, dequantize_vector
)


# ==================== 数据模型 ====================

class UpdateRequest(BaseModel):
    """客户端上传更新的请求"""
    client_id: int
    grad_vector: List[float]  # 序列化的梯度向量
    n_samples: int
    round_id: int
    metrics: Optional[Dict[str, float]] = None  # 训练指标（mAP, loss等）
    bits_up: Optional[float] = None # 客户端上传的模型比特数
    quant_params: Optional[Dict[str, Any]] = None # 量化参数 (scale, zero_point, bits)


class StatusResponse(BaseModel):
    """服务器状态响应"""
    current_round: int
    total_rounds: int
    training_done: bool
    buffered_updates: int
    clients_per_round: int
    waiting_for: int
    aggregation_mode: str # 新增聚合模式
    # 训练指标
    avg_map50: Optional[float] = None
    avg_loss: Optional[float] = None
    round_time: Optional[float] = None
    bits_down_total_round: Optional[float] = None # 服务器下发模型总比特数
    bits_up_total_round: Optional[float] = None # 客户端上传模型总比特数


# ==================== 服务器状态 ====================

class ServerState:
    """服务器全局状态"""
    
    def __init__(self, config: Config, initial_model):
        self.config = config
        self.model = initial_model
        self.global_state = initial_model.model.state_dict()
        self.last_global_state = initial_model.model.state_dict() # 用于FLQ模式下计算梯度差异

        # 训练状态
        self.current_round = 0
        self.training_done = False

        # 缓冲区
        self.update_buffer = []
        self.sample_counts = []
        self.metrics_buffer = []  # 存储每个客户端的指标
        self.bits_up_buffer = [] # 存储每个客户端上传的比特数

        # 统计信息
        self.round_start_time = None
        self.round_metrics = {}  # 每轮的平均指标
        self.total_params, self.model_size_mb = compute_model_size(self.global_state, 32)
        self.bits_down_per_round = 0 # 服务器下发模型大小（比特）

        print(f"[{self._ts()}] 📦 模型参数: {self.total_params:,} ({self.model_size_mb:.1f} MB)")
        print(f"[{self._ts()}] 🎯 训练目标: {config.rounds} 轮 × {config.clients_per_round} 客户端")
    
    def _ts(self):
        """时间戳"""
        return datetime.now().strftime('%H:%M:%S')
    
    def add_update(self, grad_vector: torch.Tensor, n_samples: int, metrics: Optional[Dict] = None, bits_up: Optional[float] = None, quant_params: Optional[Dict] = None):
        """添加客户端更新到缓冲区"""
        self.update_buffer.append({'grad_vector': grad_vector, 'quant_params': quant_params})
        self.sample_counts.append(n_samples)
        if metrics:
            self.metrics_buffer.append(metrics)
        if bits_up is not None:
            self.bits_up_buffer.append(bits_up)

        waiting = self.config.clients_per_round - len(self.update_buffer)
        print(f"[{self._ts()}] 📥 收到客户端更新 ({len(self.update_buffer)}/{self.config.clients_per_round})")

        if len(self.update_buffer) >= self.config.clients_per_round:
            self._aggregate_and_advance()
    
    def _aggregate_and_advance(self):
        """聚合更新并推进到下一轮"""
        print(f"\n{'='*70}")
        print(f"[{self._ts()}] 🔄 聚合 Round {self.current_round} (Mode: {self.config.aggregation_mode})")
        print(f"{'='*70}")

        aggregated_grad_vector = None
        if self.config.aggregation_mode == "fedavg":
            # FedAvg 聚合 state_dict
            state_dict_updates = [item['grad_vector'] for item in self.update_buffer] # 这里的grad_vector实际上是state_dict
            self.global_state = fedavg_aggregate(state_dict_updates, self.sample_counts)
        elif self.config.aggregation_mode == "flq-fed":
            # FLQ-Fed 聚合梯度向量
            grad_vectors = []
            for item in self.update_buffer:
                grad_vec = torch.tensor(item['grad_vector'])
                quant_params = item['quant_params']
                
                if quant_params:
                    # 反量化
                    dequantized_grad_vec = dequantize_vector(
                        grad_vec,
                        quant_params['scale'],
                        quant_params['zero_point'],
                        quant_params['bits']
                    )
                    grad_vectors.append(dequantized_grad_vec)
                else:
                    grad_vectors.append(grad_vec) # 如果没有量化参数，说明是全精度梯度
            
            aggregated_grad_vector = fedavg_aggregate(grad_vectors, self.sample_counts)
            self.global_state = grad_vector_to_state_dict(aggregated_grad_vector, self.last_global_state)
        else:
            raise ValueError(f"未知的聚合模式: {self.config.aggregation_mode}")

        # 更新 last_global_state
        self.last_global_state = self.global_state

        # 计算服务器下发模型大小
        _, bits_down_mb = compute_model_size(self.global_state, bits=32) # 假设是32bit浮点数
        self.bits_down_per_round = bits_down_mb * (1024 ** 2) * 8 # 转换为比特

        # 聚合客户端指标和通信量
        round_metrics = {}
        if self.metrics_buffer:
            for key in self.metrics_buffer[0].keys():
                values = [m.get(key, 0.0) for m in self.metrics_buffer]
                round_metrics[key] = sum(values) / len(values)
        
        if self.bits_up_buffer:
            round_metrics['bits_up_total_round'] = sum(self.bits_up_buffer)
        round_metrics['bits_down_total_round'] = self.bits_down_per_round

        self.round_metrics[self.current_round] = round_metrics
        print(f"📊 平均指标: mAP50={round_metrics.get('map50', 0.0):.4f}, Loss={round_metrics.get('loss', 0.0):.4f}")
        print(f"⬆️  本轮上传总比特: {round_metrics.get('bits_up_total_round', 0.0):.2f}")
        print(f"⬇️  本轮下发总比特: {round_metrics.get('bits_down_total_round', 0.0):.2f}")

        # 保存checkpoint
        save_dir = Path(self.config.server_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"global_round_{self.current_round + 1}.pt"
        torch.save(self.global_state, checkpoint_path)

        # 统计信息
        round_time = (datetime.now() - self.round_start_time).total_seconds() if self.round_start_time else 0
        compress_ratio = compute_compression_ratio(32, self.config.quant_bits if self.config.quant_enabled else 32)

        print(f"⏱️  轮次时间: {round_time:.1f}s")
        print(f"💾 Checkpoint: {checkpoint_path}")
        print(f"🗜️  压缩率: {compress_ratio:.2f}x")
        print(f"{'='*70}\n")

        # 清空缓冲区并推进
        self.update_buffer.clear()
        self.sample_counts.clear()
        self.metrics_buffer.clear()
        self.bits_up_buffer.clear()
        self.current_round += 1
        self.round_start_time = datetime.now()

        # 检查是否完成
        if self.current_round >= self.config.rounds:
            self.training_done = True
            print(f"🎉 所有训练轮次已完成！")
            self._save_metrics_to_csv()
    
    def get_global_model(self) -> tuple:
        """获取全局模型"""
        return self.global_state, self.current_round, self.training_done

    def _save_metrics_to_csv(self):
        """保存训练指标到CSV文件"""
        import csv

        csv_path = Path(self.config.server_save_dir) / "training_metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 动态生成表头
            if not self.round_metrics:
                writer.writerow(['round'])
                return
            
            first_round_metrics = next(iter(self.round_metrics.values()))
            fieldnames = ['round'] + sorted(first_round_metrics.keys())
            writer.writerow(fieldnames)

            for round_id, metrics in self.round_metrics.items():
                row = [round_id] + [metrics.get(key, '') for key in sorted(first_round_metrics.keys())]
                writer.writerow(row)

        print(f"\n📊 训练指标已保存到: {csv_path}")

    def get_current_metrics(self) -> Dict:
        """获取当前轮次的指标"""
        if self.current_round > 0 and (self.current_round - 1) in self.round_metrics:
            return self.round_metrics[self.current_round - 1]
        return {}


# ==================== FastAPI 应用 ====================

def create_app(config: Config, initial_model) -> FastAPI:
    """创建 FastAPI 应用"""
    
    app = FastAPI(title="FLQ-Fed Server")
    state = ServerState(config, initial_model)
    
    @app.get("/")
    def root():
        return {"message": "FLQ-Fed Server", "version": "2.0-simplified"}
    
    @app.get("/status", response_model=StatusResponse)
    def get_status():
        """获取服务器状态"""
        current_metrics = state.get_current_metrics()
        round_time = (datetime.now() - state.round_start_time).total_seconds() if state.round_start_time else 0.0

        return StatusResponse(
            current_round=state.current_round,
            total_rounds=state.config.rounds,
            training_done=state.training_done,
            buffered_updates=len(state.update_buffer),
            clients_per_round=state.config.clients_per_round,
            waiting_for=state.config.clients_per_round - len(state.update_buffer),
            aggregation_mode=state.config.aggregation_mode,
            avg_map50=current_metrics.get('map50'),
            avg_loss=current_metrics.get('loss'),
            round_time=round_time,
            bits_down_total_round=current_metrics.get('bits_down_total_round'),
            bits_up_total_round=current_metrics.get('bits_up_total_round')
        )
    
    @app.get("/global")
    def get_global():
        """客户端拉取全局模型"""
        global_state, round_id, done = state.get_global_model()
        
        # 如果启用下行量化
        if state.config.downlink_quant_bits > 0:
            # 将 state_dict 转换为向量
            global_vector = state_dict_to_vector(global_state)
            
            # 量化
            quantized_vector, scale, zero_point = quantize_vector(
                global_vector, bits=state.config.downlink_quant_bits
            )
            
            # 反量化回全精度，以便客户端直接加载
            dequantized_vector = dequantize_vector(
                quantized_vector, scale, zero_point, bits=state.config.downlink_quant_bits
            )
            
            # 转换回 state_dict
            global_state = vector_to_state_dict(dequantized_vector, global_state)
            
            # 计算下发模型大小（量化后）
            _, bits_down_mb = compute_model_size(global_state, bits=state.config.downlink_quant_bits)
            state.bits_down_per_round = bits_down_mb * (1024 ** 2) * 8
        else:
            # 否则，按32bit计算
            _, bits_down_mb = compute_model_size(global_state, bits=32)
            state.bits_down_per_round = bits_down_mb * (1024 ** 2) * 8

        # 序列化 state_dict
        serialized = {k: v.cpu().tolist() for k, v in global_state.items()}
        
        return {
            "state_dict": serialized,
            "round": round_id,
            "done": done,
            "downlink_quant_bits": state.config.downlink_quant_bits # 告知客户端下行量化比特数
        }
    
    @app.post("/update")
    def receive_update(request: UpdateRequest):
        """接收客户端更新"""
        if state.training_done:
            return {"success": True, "message": "训练已完成", "done": True}

        # 反序列化 grad_vector
        grad_vector = torch.tensor(request.grad_vector)

        # 添加更新（包括指标和比特数）
        state.add_update(
            grad_vector,
            request.n_samples,
            request.metrics,
            request.bits_up,
            request.quant_params
        )

        return {
            "success": True,
            "round": state.current_round,
            "done": state.training_done,
            "buffered": len(state.update_buffer)
        }
    
    return app


# ==================== 启动函数 ====================

def start_server(config_path: Optional[str] = None):
    """
    启动联邦学习服务器
    
    Args:
        config_path: 配置文件路径（可选）
    """
    print("="*70)
    print("🚀 FLQ-Fed 联邦学习服务器")
    print("="*70)
    
    # 加载配置
    config = Config(config_path)
    print(f"✅ 配置: {config}\n")
    
    # 初始化模型
    from ultralytics import YOLO
    project_root = Path(__file__).parent.parent
    model_path = project_root / config.model_name
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📦 加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    # 初始化类别数（从数据配置读取）
    data_yaml = project_root / "data" / "client1" / "oil.yaml"
    if data_yaml.exists():
        import yaml
        with open(data_yaml) as f:
            data_cfg = yaml.safe_load(f)
        nc = data_cfg.get('nc', 80)
        
        from ultralytics.nn.tasks import DetectionModel
        model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 模型初始化完成 (nc={nc})\n")
    
    # 创建 FastAPI 应用
    app = create_app(config, model)
    
    # 启动服务器
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌐 启动服务器: http://{config.server_host}:{config.server_port}")
    print(f"{'='*70}\n")
    
    uvicorn.run(
        app,
        host=config.server_host,
        port=config.server_port,
        log_level="warning"
    )


if __name__ == "__main__":
    start_server()

