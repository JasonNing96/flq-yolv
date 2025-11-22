"""
FLQ-Fed 模型工具函数
整合量化、聚合、模型转换等核心功能
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
    """将 state_dict 展平为向量"""
    return torch.cat([v.flatten() for v in state_dict.values()])


def vector_to_state_dict(vector: torch.Tensor, template: Dict) -> Dict:
    """将向量恢复为 state_dict"""
    state_dict = {}
    pointer = 0
    for key, value in template.items():
        num_param = value.numel()
        state_dict[key] = vector[pointer:pointer + num_param].view_as(value)
        pointer += num_param
    return state_dict


def state_dict_to_grad_vector(client_state_dict: Dict, global_state_dict: Dict) -> torch.Tensor:
    """
    计算客户端模型与全局模型之间的梯度差异，并展平为一维向量。
    """
    grad_list = []
    for key in global_state_dict.keys():
        grad_list.append((client_state_dict[key] - global_state_dict[key]).flatten())
    return torch.cat(grad_list)


def grad_vector_to_state_dict(grad_vector: torch.Tensor, global_state_dict: Dict) -> Dict:
    """
    将一维梯度向量应用到全局 state_dict 上，生成新的 state_dict。
    """
    new_state_dict = {}
    pointer = 0
    for key, value in global_state_dict.items():
        num_param = value.numel()
        grad_param = grad_vector[pointer:pointer + num_param].view_as(value)
        new_state_dict[key] = value + grad_param
        pointer += num_param
    return new_state_dict


# ==================== 量化函数 ====================

def quantize_vector(vector: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
    """
    量化向量到指定比特数
    
    Args:
        vector: 输入向量
        bits: 量化比特数 (1, 4, 8)
    
    Returns:
        quantized: 量化后的向量
        scale: 缩放因子
        zero_point: 零点
    """
    if bits == 32:  # FP32，无需量化
        return vector, 1.0, 0.0
    
    # 计算范围
    min_val = vector.min().item()
    max_val = vector.max().item()
    
    if bits == 1:  # 1-bit: 符号量化
        scale = max(abs(min_val), abs(max_val))
        quantized = torch.sign(vector)
        zero_point = 0.0
    else:  # 4-bit 或 8-bit
        levels = 2 ** bits
        scale = (max_val - min_val) / (levels - 1)
        zero_point = min_val
        
        if scale == 0:  # 避免除零
            return torch.zeros_like(vector), 0.0, 0.0
        
        quantized = torch.round((vector - zero_point) / scale)
        quantized = torch.clamp(quantized, 0, levels - 1)
    
    return quantized, scale, zero_point


def dequantize_vector(quantized: torch.Tensor, scale: float, zero_point: float, bits: int = 8) -> torch.Tensor:
    """反量化向量"""
    if bits == 32:
        return quantized
    elif bits == 1:
        return quantized * scale
    else:
        return quantized * scale + zero_point


# ==================== 聚合函数 ====================

def fedavg_aggregate(updates: List[torch.Tensor], weights: List[int]) -> torch.Tensor:
    """
    FedAvg 聚合算法（加权平均）
    
    Args:
        updates: 客户端更新列表 [grad_vector, ...]
        weights: 客户端样本数列表 [n1, n2, ...]
    
    Returns:
        aggregated: 聚合后的 grad_vector
    """
    if not updates:
        raise ValueError("没有可聚合的更新")
    
    # 归一化权重
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # 加权平均
    aggregated_vector = torch.zeros_like(updates[0])
    for w, update_vector in zip(normalized_weights, updates):
        aggregated_vector += w * update_vector
    
    return aggregated_vector


# ==================== 统计函数 ====================

def compute_model_size(state_dict: Dict, bits: int = 32) -> Tuple[int, float]:
    """
    计算模型大小（参数数量和字节数）
    
    Args:
        state_dict: 模型状态字典
        bits: 量化比特数
    
    Returns:
        num_params: 参数总数
        size_mb: 模型大小（MB）
    """
    num_params = sum(p.numel() for p in state_dict.values())
    size_bytes = num_params * bits / 8
    size_mb = size_bytes / (1024 ** 2)
    
    return num_params, size_mb


def compute_compression_ratio(original_bits: int, compressed_bits: int) -> float:
    """计算压缩率"""
    return original_bits / compressed_bits if compressed_bits > 0 else 1.0


# ==================== 误差反馈（可选）====================

class ErrorFeedback:
    """误差反馈机制（用于1-bit量化）"""
    
    def __init__(self):
        self.error = None
    
    def compress_with_feedback(self, vector: torch.Tensor, bits: int = 1) -> Tuple[torch.Tensor, float, float]:
        """带误差反馈的压缩"""
        # 如果有累积误差，先加上
        if self.error is not None:
            vector = vector + self.error
        
        # 量化
        quantized, scale, zero_point = quantize_vector(vector, bits)
        
        # 反量化
        dequantized = dequantize_vector(quantized, scale, zero_point, bits)
        
        # 计算并存储误差
        self.error = vector - dequantized
        
        return quantized, scale, zero_point
    
    def reset(self):
        """重置误差"""
        self.error = None

