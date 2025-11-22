"""
FLQ-Fed 配置管理
简化的配置加载和访问
"""
import yaml
from pathlib import Path
from typing import Optional


class Config:
    """配置类 - 简化访问"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径，默认为 configs/flq_config.yaml
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "configs" / "flq_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    # ===== 训练参数 =====
    @property
    def rounds(self) -> int:
        return self._config['training']['rounds']
    
    @property
    def clients_per_round(self) -> int:
        return self._config['training']['clients_per_round']
    
    @property
    def local_epochs(self) -> int:
        return self._config['training']['local_epochs']
    
    # ===== 量化参数 =====
    @property
    def quant_enabled(self) -> bool:
        return self._config['quantization']['enabled']
    
    @property
    def quant_bits(self) -> int:
        return self._config['quantization']['bits']
    
    @property
    def error_feedback(self) -> bool:
        return self._config['quantization'].get('use_error_feedback', False)

    @property
    def error_feedback_enabled(self) -> bool:
        return self._config['quantization'].get('error_feedback_enabled', False)
    
    # ===== 模型参数 =====
    @property
    def model_name(self) -> str:
        return self._config['model']['name']
    
    @property
    def device(self) -> str:
        return self._config['model']['device']
    
    # ===== 服务器参数 =====
    @property
    def server_host(self) -> str:
        return self._config['server']['host']
    
    @property
    def server_port(self) -> int:
        return self._config['server']['port']
    
    @property
    def server_save_dir(self) -> str:
        return self._config['server']['save_dir']

    @property
    def aggregation_mode(self) -> str:
        return self._config['server'].get('aggregation_mode', 'fedavg')

    @property
    def downlink_quant_bits(self) -> int:
        return self._config['server'].get('downlink_quant_bits', 0)
    
    # ===== 客户端参数 =====
    @property
    def batch_size(self) -> int:
        return self._config['client']['batch_size']
    
    @property
    def workers(self) -> int:
        return self._config['client'].get('workers', 0)
    
    @property
    def verbose(self) -> bool:
        return self._config['client'].get('verbose', True)
    
    @property
    def enable_val(self) -> bool:
        return self._config['client'].get('enable_val', False)
    
    @property
    def enable_plots(self) -> bool:
        return self._config['client'].get('enable_plots', False)
    
    def __str__(self):
        """友好的字符串表示"""
        quant_str = f"{self.quant_bits}bit" if self.quant_enabled else "OFF"
        ef_str = "EF" if self.error_feedback_enabled else "NoEF"
        down_quant_str = f"{self.downlink_quant_bits}bit" if self.downlink_quant_bits > 0 else "FP32"
        return (f"FLQConfig(rounds={self.rounds}, clients={self.clients_per_round}, "
                f"quant={quant_str} ({ef_str}), agg_mode={self.aggregation_mode}, "
                f"down_quant={down_quant_str}, device={self.device})")

