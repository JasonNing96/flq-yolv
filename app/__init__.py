"""
FLQ-Fed 核心模块
精简化联邦学习框架
"""

__version__ = "2.0-simplified"

from .config import Config
from .model_utils import (
    model_to_vector,
    vector_to_model,
    state_dict_to_vector,
    vector_to_state_dict,
    quantize_vector,
    dequantize_vector,
    fedavg_aggregate,
    compute_model_size,
    compute_compression_ratio,
)

__all__ = [
    'Config',
    'model_to_vector',
    'vector_to_model',
    'state_dict_to_vector',
    'vector_to_state_dict',
    'quantize_vector',
    'dequantize_vector',
    'fedavg_aggregate',
    'compute_model_size',
    'compute_compression_ratio',
]

