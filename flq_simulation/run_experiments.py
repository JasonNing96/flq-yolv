import yaml
import subprocess
import time
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "flq_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def update_config_file(config_data: dict):
    """更新 flq_config.yaml 文件"""
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_data, f, sort_keys=False)
    print(f"Updated config file: {CONFIG_PATH}")

def run_single_experiment(experiment_name: str, config_updates: dict, num_clients: int = 6):
    """运行单个实验并保存结果"""
    print(f"\n{'='*80}")
    print(f"Starting experiment: {experiment_name}")
    print(f"{'='*80}")

    # 读取基础配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    # 应用实验特定的更新
    # 注意：这里需要深拷贝，以防修改到原始的base_config
    current_config = base_config.copy()
    for key, value in config_updates.items():
        if key in current_config and isinstance(current_config[key], dict) and isinstance(value, dict):
            current_config[key].update(value)
        else:
            current_config[key] = value
    
    # 确保客户端数量正确
    if 'training' in current_config:
        current_config['training']['clients_per_round'] = num_clients

    # 更新配置文件
    update_config_file(current_config)

    # 定义输出 CSV 路径
    csv_output_path = OUTPUT_DIR / f"{experiment_name}.csv"

    # 运行 fed_flq_local.py
    command = [
        "python",
        str(PROJECT_ROOT / "app" / "fed_flq_local.py"),
        "train",
        "--config",
        str(CONFIG_PATH),
        "--csv",
        str(csv_output_path),
        "--num-clients",
        str(num_clients)
    ]
    
    print(f"Running command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Experiment '{experiment_name}' failed!")
        print("Stdout:", process.stdout)
        print("Stderr:", process.stderr)
    else:
        print(f"Experiment '{experiment_name}' completed successfully. Results saved to {csv_output_path}")
        print("Stdout:", process.stdout)
    
    time.sleep(5) # 等待进程完全关闭

def main():
    # 定义实验列表
    experiments = [
        {
            "name": "FedAvg_FP32",
            "config_updates": {
                "quantization": {"enabled": False, "bits": 32, "error_feedback_enabled": False},
                "server": {"aggregation_mode": "fedavg", "downlink_quant_bits": 0}
            }
        },
        {
            "name": "FLQ_1bit_EF",
            "config_updates": {
                "quantization": {"enabled": True, "bits": 1, "error_feedback_enabled": True},
                "server": {"aggregation_mode": "flq-fed", "downlink_quant_bits": 0}
            }
        },
        {
            "name": "FLQ_4bit_NoEF",
            "config_updates": {
                "quantization": {"enabled": True, "bits": 4, "error_feedback_enabled": False},
                "server": {"aggregation_mode": "flq-fed", "downlink_quant_bits": 0}
            }
        },
        {
            "name": "FLQ_8bit_NoEF",
            "config_updates": {
                "quantization": {"enabled": True, "bits": 8, "error_feedback_enabled": False},
                "server": {"aggregation_mode": "flq-fed", "downlink_quant_bits": 0}
            }
        },
    ]

    for exp in experiments:
        run_single_experiment(exp["name"], exp["config_updates"], num_clients=6)

    print("\nAll experiments finished. You can now plot the results.")

if __name__ == "__main__":
    main()
