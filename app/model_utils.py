"""
FLQ-Fed 模型工具函数
整合量化、聚合、模型转换等核心功能
移植自 flq_yolov_v7.py (FLQ-YOLOv7 Server Momentum Version)
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

# ==================== 模型 ↔ 向量转换 ====================

def model_to_vector(model) -> torch.Tensor:
    """将模型参数展平为一维向量"""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def vector_to_model(vector: torch.Tensor, model):
    """将一维向量恢复为模型参数"""
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = vector[pointer:pointer + num_param].view_as(param).data
        pointer += num_param


def state_dict_to_vector(state_dict: Dict) -> torch.Tensor:
    """
    将 state_dict 展平为向量
    注意：只处理 float 类型的参数，忽略整数参数（如 batch_norm.num_batches_tracked）
    """
    tensors = [v.flatten() for v in state_dict.values() if v.dtype.is_floating_point]
    if not tensors:
        return torch.tensor([])
    return torch.cat(tensors)


def vector_to_state_dict(vector: torch.Tensor, template: Dict) -> Dict:
    """
    将向量恢复为 state_dict
    注意：只恢复 float 类型的参数，其他参数直接从 template 复制
    """
    state_dict = {}
    pointer = 0
    device = vector.device
    
    for key, value in template.items():
        if value.dtype.is_floating_point:
            num_param = value.numel()
            # 确保形状匹配
            state_dict[key] = vector[pointer:pointer + num_param].view_as(value).to(value.dtype)
            pointer += num_param
        else:
            # 对于非浮点参数（如 int64 的计数器），直接使用 template 中的值
            state_dict[key] = value.clone()
            
    return state_dict


def state_dict_to_grad_vector(client_state_dict: Dict, global_state_dict: Dict) -> torch.Tensor:
    """
    计算客户端模型与全局模型之间的梯度差异，并展平为一维向量。
    grad = local - global
    """
    grad_list = []
    # 确保 key 顺序一致
    for key in global_state_dict.keys():
        v_global = global_state_dict[key]
        if v_global.dtype.is_floating_point:
            v_local = client_state_dict[key]
            # 确保都在同一设备
            grad_list.append((v_local - v_global).flatten())
            
    if not grad_list:
        return torch.tensor([])
    return torch.cat(grad_list)


# ==================== FLQ 压缩器 (移植自 v7) ====================

class FLQCompressor:
    """
    FLQ 压缩器，支持 Error Feedback
    """
    def __init__(self, device: str = 'cpu'):
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

    def quantize_update(self, delta_vec: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int, float, float]:
        """
        量化更新向量
        Returns:
            quantized_delta: 量化后的向量
            bit_cost: 传输比特数
            scale: 缩放因子
            zero_point: 零点 (v7实现中 zero_point 是通过 min/max 计算的 bias)
        """
        num_params = delta_vec.numel()
        
        # 转移到设备
        delta_vec = delta_vec.to(self.device)
        
        if self.local_error is None:
            self.local_error = torch.zeros_like(delta_vec)
        else:
            self.local_error = self.local_error.to(self.device)
            
        target = delta_vec + self.local_error

        if bits >= 32:
            self.local_error.zero_()
            return target, num_params * 32, 1.0, 0.0

        if bits == 1:
            scale = target.abs().mean()
            if scale < 1e-8: scale = 1e-8
            sign = torch.sign(target)
            sign[sign == 0] = 1.0
            quantized = sign * scale
            self.local_error = target - quantized
            # 1-bit 传输开销：每个参数1bit + scale(32bit)
            # 这里简化返回 scale, zero_point=0
            return sign, num_params + 32, scale.item(), 0.0
        else:
            mn, mx = target.min(), target.max()
            scale = (mx - mn) / (2**bits - 1 + 1e-8)
            zero = -mn / (scale + 1e-8)
            
            q = torch.clamp(torch.round(target / scale + zero), 0, 2**bits - 1)
            dq = (q - zero) * scale
            
            self.local_error = target - dq
            
            # 返回量化后的整数 q，以及反量化参数
            return q, num_params * bits + 64, scale.item(), zero.item()

    @staticmethod
    def dequantize(quantized: torch.Tensor, scale: float, zero: float, bits: int) -> torch.Tensor:
        """反量化"""
        if bits >= 32:
            return quantized
        if bits == 1:
            return quantized * scale
        
        # q -> dq = (q - zero) * scale
        return (quantized - zero) * scale


# ==================== 聚合函数 ====================

def fedavg_aggregate(updates: List[torch.Tensor], weights: List[int]) -> torch.Tensor:
    """
    FedAvg 聚合算法（加权平均）
    """
    if not updates:
        raise ValueError("没有可聚合的更新")
    
    # 归一化权重
    total_weight = sum(weights)
    if total_weight <= 0:
        normalized_weights = [1.0 / len(weights)] * len(weights)
    else:
        normalized_weights = [w / total_weight for w in weights]
    
    # 确保所有 updates 在同一设备
    device = updates[0].device
    
    # 加权平均
    aggregated_vector = torch.zeros_like(updates[0], device=device)
    for w, update_vector in zip(normalized_weights, updates):
        aggregated_vector += w * update_vector.to(device)
    
    return aggregated_vector


# ==================== 统计函数 ====================

def compute_model_size(state_dict: Dict, bits: int = 32) -> Tuple[int, float]:
    """
    计算模型大小（参数数量和字节数）
    """
    num_params = sum(p.numel() for p in state_dict.values())
    size_bytes = num_params * bits / 8
    size_mb = size_bytes / (1024 ** 2)
    
    return num_params, size_mb


def compute_compression_ratio(original_bits: int, compressed_bits: int) -> float:
    """计算压缩率"""
    return original_bits / compressed_bits if compressed_bits > 0 else 1.0
