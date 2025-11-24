"""
FLQ-Fed 联邦学习服务器
简化版 - 集中所有服务器逻辑
匹配 flq_yolov_v8.py (FreezeBN Version)
"""
import os
import torch
import gc
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
    state_dict_to_grad_vector,
    FLQCompressor # 使用统一的压缩器类
)


# ==================== 数据模型 ====================

class UpdateRequest(BaseModel):
    """客户端上传更新的请求"""
    client_id: int
    grad_vector: List[float]  # 序列化的梯度向量 (或量化后的整数向量)
    n_samples: int
    round_id: int
    metrics: Optional[Dict[str, float]] = None
    bits_up: Optional[float] = None
    quant_params: Optional[Dict[str, Any]] = None # {scale, zero_point, bits}


class StatusResponse(BaseModel):
    """服务器状态响应"""
    current_round: int
    total_rounds: int
    training_done: bool
    buffered_updates: int
    clients_per_round: int
    waiting_for: int
    aggregation_mode: str
    avg_map50: Optional[float] = None
    avg_loss: Optional[float] = None
    round_time: Optional[float] = None
    bits_down_total_round: Optional[float] = None
    bits_up_total_round: Optional[float] = None


# ==================== 服务器状态 ====================

class ServerState:
    """服务器全局状态"""
    
    def __init__(self, config: Config, initial_model):
        self.config = config
        self.model = initial_model
        # 保持 state_dict 在 CPU 以节省显存
        self.global_state = {k: v.cpu() for k, v in initial_model.model.state_dict().items()}
        self.last_global_state = {k: v.clone() for k, v in self.global_state.items()}

        # 训练状态
        self.current_round = 0
        self.training_done = False

        # 缓冲区
        self.update_buffer = []
        self.sample_counts = []
        self.metrics_buffer = []
        self.bits_up_buffer = []

        # 统计信息
        self.round_start_time = None
        self.round_metrics = {}
        self.total_params, self.model_size_mb = compute_model_size(self.global_state, 32)
        self.bits_down_per_round = 0

        print(f"[{self._ts()}] 📦 模型参数: {self.total_params:,} ({self.model_size_mb:.1f} MB)")
        print(f"[{self._ts()}] 🎯 训练目标: {config.rounds} 轮 × {config.clients_per_round} 客户端")
        print(f"[{self._ts()}] ⚙️  模式: {config.aggregation_mode} (v8 FreezeBN logic)")
    
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
        print(f"[{self._ts()}] 🔄 聚合 Round {self.current_round}")
        print(f"{'='*70}")

        if self.config.aggregation_mode == "flq-fed" or self.config.aggregation_mode == "fedavg":
            # 统一处理：解压 -> 聚合 -> 更新
            grad_vectors = []
            
            for item in self.update_buffer:
                vec = item['grad_vector'] # Tensor
                q_params = item['quant_params']
                
                if q_params:
                    # 反量化
                    dequantized = FLQCompressor.dequantize(
                        vec,
                        q_params['scale'],
                        q_params['zero_point'],
                        q_params['bits']
                    )
                    grad_vectors.append(dequantized)
                else:
                    grad_vectors.append(vec)
            
            # 聚合梯度差异 (avg_update)
            # v8 逻辑: Global LR = 1.0 (无 momentum)
            # global_new = global_old + 1.0 * avg(local - global)
            aggregated_grad = fedavg_aggregate(grad_vectors, self.sample_counts)
            
            # 更新全局模型
            # state_dict_new = state_dict_old + aggregated_grad
            # 注意：需将 aggregated_grad 映射回 state_dict 结构
            
            # 先展平 old global
            flat_global = state_dict_to_vector(self.global_state) # 只包含 float 参数
            
            # 更新
            flat_new_global = flat_global + aggregated_grad.cpu()
            
            # 重构 state_dict
            self.global_state = vector_to_state_dict(flat_new_global, self.global_state)
            
            # 清理
            del grad_vectors, aggregated_grad, flat_global, flat_new_global
            gc.collect()

        else:
            print(f"⚠️  未知的聚合模式: {self.config.aggregation_mode}, 跳过聚合")

        # 更新 last_global_state (用于下轮)
        # self.last_global_state = {k: v.clone() for k, v in self.global_state.items()}

        # 计算通信量
        _, bits_down_mb = compute_model_size(self.global_state, bits=32)
        self.bits_down_per_round = bits_down_mb * (1024 ** 2) * 8

        # 聚合指标
        round_metrics = {}
        if self.metrics_buffer:
            # 简单的数值平均
            keys = set()
            for m in self.metrics_buffer: keys.update(m.keys())
            
            for key in keys:
                values = [m.get(key, 0.0) for m in self.metrics_buffer]
                round_metrics[key] = sum(values) / len(values)
        
        if self.bits_up_buffer:
            round_metrics['bits_up_total_round'] = sum(self.bits_up_buffer)
        round_metrics['bits_down_total_round'] = self.bits_down_per_round

        self.round_metrics[self.current_round] = round_metrics
        print(f"📊 平均 Loss: {round_metrics.get('final_loss', 0.0):.4f}")
        print(f"⬆️  本轮上传总比特: {round_metrics.get('bits_up_total_round', 0.0) / 1e6:.2f} Mb")

        # 保存checkpoint
        save_dir = Path(self.config.server_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if (self.current_round + 1) % 5 == 0:
            checkpoint_path = save_dir / f"global_round_{self.current_round + 1}.pt"
            torch.save(self.global_state, checkpoint_path)
            print(f"💾 Checkpoint: {checkpoint_path}")

        # 统计时间
        round_time = (datetime.now() - self.round_start_time).total_seconds() if self.round_start_time else 0
        print(f"⏱️  轮次时间: {round_time:.1f}s")
        print(f"{'='*70}\n")

        # 重置缓冲区
        self.update_buffer.clear()
        self.sample_counts.clear()
        self.metrics_buffer.clear()
        self.bits_up_buffer.clear()
        
        self.current_round += 1
        self.round_start_time = datetime.now()

        if self.current_round >= self.config.rounds:
            self.training_done = True
            print(f"🎉 所有训练轮次已完成！")
            self._save_metrics_to_csv()
    
    def get_global_model(self) -> tuple:
        """获取全局模型"""
        return self.global_state, self.current_round, self.training_done

    def _save_metrics_to_csv(self):
        import csv
        csv_path = Path(self.config.server_save_dir) / "training_metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            if not self.round_metrics:
                writer.writerow(['round'])
                return
            first = next(iter(self.round_metrics.values()))
            headers = ['round'] + sorted(first.keys())
            writer.writerow(headers)
            for r, m in self.round_metrics.items():
                writer.writerow([r] + [m.get(k, '') for k in sorted(first.keys())])
        print(f"\n📊 训练指标已保存到: {csv_path}")

    def get_current_metrics(self) -> Dict:
        if self.current_round > 0 and (self.current_round - 1) in self.round_metrics:
            return self.round_metrics[self.current_round - 1]
        return {}


# ==================== FastAPI 应用 ====================

def create_app(config: Config, initial_model) -> FastAPI:
    app = FastAPI(title="FLQ-Fed Server v8")
    state = ServerState(config, initial_model)
    
    @app.get("/")
    def root():
        return {"message": "FLQ-Fed Server", "version": "v8-FreezeBN"}
    
    @app.get("/status", response_model=StatusResponse)
    def get_status():
        curr = state.get_current_metrics()
        rt = (datetime.now() - state.round_start_time).total_seconds() if state.round_start_time else 0.0
        return StatusResponse(
            current_round=state.current_round,
            total_rounds=state.config.rounds,
            training_done=state.training_done,
            buffered_updates=len(state.update_buffer),
            clients_per_round=state.config.clients_per_round,
            waiting_for=state.config.clients_per_round - len(state.update_buffer),
            aggregation_mode=state.config.aggregation_mode,
            avg_map50=curr.get('map50'),
            avg_loss=curr.get('final_loss'),
            round_time=rt,
            bits_down_total_round=curr.get('bits_down_total_round'),
            bits_up_total_round=curr.get('bits_up_total_round')
        )
    
    @app.get("/global")
    def get_global():
        gs, rid, done = state.get_global_model()
        # 序列化 state_dict (转为 list 以便 JSON 传输)
        serialized = {k: v.tolist() for k, v in gs.items()}
        return {
            "state_dict": serialized,
            "round": rid,
            "done": done
        }
    
    @app.post("/update")
    def receive_update(request: UpdateRequest):
        if state.training_done:
            return {"success": True, "done": True}
        
        # 转换 grad_vector 为 tensor
        grad_vector = torch.tensor(request.grad_vector)
        
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


def start_server(config_path: Optional[str] = None):
    print("="*70)
    print("🚀 FLQ-Fed 联邦学习服务器 (v8)")
    print("="*70)
    
    config = Config(config_path)
    print(f"✅ 配置: {config}\n")
    
    # 初始化模型
    from ultralytics import YOLO
    project_root = Path(__file__).parent.parent
    model_path = project_root / config.model_name
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📦 加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    # 初始化结构 (NC)
    data_yaml = project_root / "data" / "client1" / "oil.yaml"
    if data_yaml.exists():
        import yaml
        with open(data_yaml) as f:
            data_cfg = yaml.safe_load(f)
        nc = data_cfg.get('nc', 80)
        from ultralytics.nn.tasks import DetectionModel
        model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 模型初始化完成 (nc={nc})\n")
    
    app = create_app(config, model)
    
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
