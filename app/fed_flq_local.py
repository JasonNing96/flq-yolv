"""
单机单 server 多 client 并行的 Fed-FLQ 启动与分析脚本

功能：
1. 在本机启动 FLQ 服务器（后台线程）。
2. 启动多个 client 进程 (python -m app.client <client_id>) 并行训练。
3. 周期性调用 server 的 /status 接口，将每轮状态写入 CSV。
4. 提供 plot_flq() 函数，从 CSV 读取数据并绘制收敛与通信开销曲线。

注意：
- 假定已有 app.server.start_server 与 app.client.start_client 实现；
- 假定 server 提供 GET /status 接口，至少包含 current_round / max_rounds / is_finished 等字段；
- 如果你在 server 的 /status 中加入 mAP、bits_up/bits_down 等字段，本脚本会自动记录到 CSV 并用于绘图。
"""
import argparse
import csv
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 项目根目录，保证可以 import app.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests  # 轮询 /status
import matplotlib.pyplot as plt  # plot_flq 使用


# ==================== 工具函数 ====================

def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """将嵌套 dict 展平成一层，用 . 连接 key，便于写 CSV"""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_dict(v, prefix=key))
        else:
            flat[key] = v
    return flat


def write_csv_header(csv_path: Path, fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(csv_path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


# ==================== 服务器启动 ====================

def _run_server(config_path: Optional[str] = None) -> None:
    """直接调用 app.server.start_server，在后台线程中运行"""
    from app.server import start_server
    # start_server 内部是阻塞的（uvicorn.run），放在线程中即可
    if config_path:
        start_server(config_path)
    else:
        start_server()


def start_server_in_background(config_path: Optional[str] = None) -> threading.Thread:
    """在后台线程启动 server，返回线程对象"""
    th = threading.Thread(target=_run_server, args=(config_path,), daemon=True)
    th.start()
    return th


# ==================== /status 监控与 CSV 记录 ====================

def monitor_status_to_csv(
    server_url: str,
    csv_path: Path,
    poll_interval: float = 2.0,
    verbose: bool = True,
) -> None:
    """
    周期性访问 /status，将每一轮的状态记录到 CSV。

    约定：
    - /status 返回 JSON，至少包含：
        current_round: int
        max_rounds: int (可选)
        is_finished: bool (可选)
      如再包含 metrics / bits_up / bits_down 等字段，会自动展平写入 CSV。
    """
    url = server_url.rstrip("/") + "/status"

    # 等待 server 启动
    if verbose:
        print(f"[monitor] 等待服务器 {url} 启动 ...")
    while True:
        try:
            resp = requests.get(url, timeout=3.0)
            resp.raise_for_status()
            status = resp.json()
            break
        except Exception:
            time.sleep(1.0)

    flat = flatten_dict(status)
    flat["timestamp"] = time.time()

    # 确定轮次字段
    round_key_candidates = ["round", "current_round", "completed_rounds"]
    round_key = None
    for k in round_key_candidates:
        if k in flat:
            round_key = k
            break
    if round_key is None:
        raise RuntimeError(
            "status 中没有发现轮次字段（round/current_round/completed_rounds），"
            "请在 server 的 /status 中加入至少一个字段。"
        )

    # 初始化 CSV
    fieldnames = sorted(flat.keys())
    write_csv_header(csv_path, fieldnames)
    append_csv_row(csv_path, fieldnames, flat)
    last_round = flat.get(round_key, 0)

    if verbose:
        print(f"[monitor] 已连接服务器，监控轮次字段: {round_key}")
        print(f"[monitor] 首条状态: round={last_round}")

    # 不断轮询
    while True:
        time.sleep(poll_interval)
        try:
            resp = requests.get(url, timeout=3.0)
            resp.raise_for_status()
            status = resp.json()
        except Exception:
            if verbose:
                print("[monitor] 访问 /status 失败，重试中 ...")
            continue

        flat = flatten_dict(status)
        flat["timestamp"] = time.time()
        current_round = flat.get(round_key, last_round)

        # 每当轮次变化时写一行
        if current_round != last_round:
            append_csv_row(csv_path, fieldnames, flat)
            last_round = current_round
            if verbose:
                print(f"[monitor] 记录一条状态: {round_key}={current_round}")

        # 结束条件：兼容多种字段名
        # - 旧约定：is_finished / finished / max_rounds
        # - 现有 server：training_done / total_rounds
        if status.get("is_finished") or status.get("finished") or status.get("training_done"):
            if verbose:
                print("[monitor] 检测到训练结束标志（is_finished/finished/training_done），停止监控")
            break

        # 轮次上限：优先使用 max_rounds，其次 total_rounds
        max_rounds = status.get("max_rounds") or status.get("total_rounds")
        if max_rounds is not None and current_round >= max_rounds:
            if verbose:
                print("[monitor] current_round 已达到 max_rounds/total_rounds，停止监控")
            break


# ==================== 训练入口 ====================

def run_training(
    config_path: Optional[str],
    num_clients: Optional[int],
    csv_path: Path,
    enable_monitor: bool = True,
) -> int:
    """
    启动 server + 多个 client，在本机完成 Fed-FLQ 训练，并记录 CSV。

    Args:
        config_path: 配置文件路径（可选，传给 server），例如 'configs/flq_config.yaml'
        num_clients: 并行 client 个数；如果为 None，则从 Config 中读取 clients_per_round。
        csv_path: 监控结果 CSV 输出路径。
        enable_monitor: 是否启动 /status 监控线程。
    """
    from app.config import Config
    import subprocess

    cfg = Config(config_path) if config_path else Config()
    if num_clients is None:
        num_clients = cfg.clients_per_round

    server_host = getattr(cfg, "server_host", "127.0.0.1")
    server_port = getattr(cfg, "server_port", 8087)
    server_url = f"http://{server_host}:{server_port}"

    print("=" * 70)
    print(f"[launcher] 使用配置: {cfg}")
    print(f"[launcher] 启动 server 于 {server_url}")
    print(f"[launcher] 并行 client 数量: {num_clients}")
    print("=" * 70)

    # 1) 启动 server
    server_thread = start_server_in_background(config_path)
    time.sleep(5.0)  # 给 server 一点时间完成启动

    # 2) 启动监控线程
    monitor_thread = None
    if enable_monitor:
        monitor_thread = threading.Thread(
            target=monitor_status_to_csv,
            args=(server_url, csv_path),
            kwargs={"poll_interval": 2.0, "verbose": True},
            daemon=True,
        )
        monitor_thread.start()

    # 3) 启动多个 client 子进程
    processes: List["subprocess.Popen"] = []
    try:
        for cid in range(1, num_clients + 1):
            cmd = [sys.executable, "-m", "app.client", str(cid)]
            env = os.environ.copy()
            # 如需从外部传 config，可在 app.client 中读取该环境变量
            if config_path:
                env.setdefault("FLQ_CONFIG", config_path)

            proc = subprocess.Popen(cmd, env=env)
            processes.append(proc)
            print(f"[launcher] 已启动 client #{cid}, PID={proc.pid}")
            time.sleep(1.0)  # 轻微错峰

        # 4) 等待所有 client 结束
        exit_codes = []
        for proc in processes:
            code = proc.wait()
            exit_codes.append(code)
            print(f"[launcher] client PID={proc.pid} 结束，退出码={code}")

    except KeyboardInterrupt:
        print("\n[launcher] 收到中断信号，尝试终止所有 client ...")
        for proc in processes:
            with suppress(Exception):
                proc.send_signal(signal.SIGINT)
        return 1

    # 5) 等监控线程自然结束（如果还在跑）
    if monitor_thread is not None and monitor_thread.is_alive():
        print("[launcher] 等待监控线程结束 ...")
        monitor_thread.join(timeout=10.0)

    print("[launcher] 训练流程结束")
    print(f"[launcher] CSV 已保存至: {csv_path}")
    return 0


# ==================== 绘图函数 ====================

def plot_flq(csv_paths: Dict[str, Path], out_dir: Path) -> None:
    """
    从多个 CSV 读取数据，绘制若干典型曲线进行对比：
    1) Round – mAP / Accuracy
    2) Round – bits_down / bits_up
    3) Cumulative Bits – mAP / Accuracy
    
    Args:
        csv_paths: 实验名称到 CSV 文件路径的映射。
        out_dir: 图片输出目录。
    """
    import pandas as pd
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = {}
    for name, path in csv_paths.items():
        if not path.exists():
            print(f"[plot] 警告: CSV 文件不存在，跳过: {path}")
            continue
        df = pd.read_csv(path)
        
        # 确定轮次列
        round_col = None
        for cand in ["round", "current_round", "completed_rounds"]:
            if cand in df.columns:
                round_col = cand
                break
        if round_col is None:
            print(f"[plot] 警告: CSV 文件 {path} 中未找到轮次列，跳过。")
            continue
        
        df['round_col'] = df[round_col] # 统一列名
        all_dfs[name] = df
    
    if not all_dfs:
        print("[plot] 没有有效的 CSV 数据可供绘图。")
        return

    # 1) 精度 / mAP 曲线对比
    metric_cols = []
    # 收集所有 df 中存在的 metric 列
    for df in all_dfs.values():
        metric_cols.extend([c for c in df.columns if any(k in c.lower() for k in ["map", "acc", "accuracy"])])
    metric_cols = sorted(list(set(metric_cols))) # 去重并排序

    for col in metric_cols:
        plt.figure(figsize=(10, 6))
        for name, df in all_dfs.items():
            if col in df.columns:
                plt.plot(df["round_col"], df[col], marker="o", label=name)
        
        plt.xlabel("Round")
        plt.ylabel(col)
        plt.grid(True)
        plt.legend()
        plt.title(f"{col} vs. Round Comparison")
        fig_path = out_dir / f"{col.replace('.', '_')}_vs_round_comparison.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"[plot] 保存图像: {fig_path}")

    # 2) 通信比特数曲线对比
    plt.figure(figsize=(10, 6))
    has_bits_data = False
    for name, df in all_dfs.items():
        if "bits_down_total_round" in df.columns and "bits_up_total_round" in df.columns:
            df["total_bits_per_round"] = df["bits_down_total_round"] + df["bits_up_total_round"]
            plt.plot(df["round_col"], df["total_bits_per_round"], marker="o", label=name)
            has_bits_data = True
        elif "bits_up_total_round" in df.columns: # 如果只有上传比特
            plt.plot(df["round_col"], df["bits_up_total_round"], marker="o", label=name + " (Up Only)")
            has_bits_data = True

    if has_bits_data:
        plt.xlabel("Round")
        plt.ylabel("Total Communication Bits per Round")
        plt.grid(True)
        plt.legend()
        plt.title("Total Communication Bits per Round Comparison")
        fig_path = out_dir / "total_bits_vs_round_comparison.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"[plot] 保存图像: {fig_path}")
    else:
        plt.close()
        print("[plot] 没有足够的通信比特数据可供绘图。")

    # 3) 累计比特 – 精度曲线对比
    if has_bits_data and metric_cols:
        for col in metric_cols:
            plt.figure(figsize=(10, 6))
            plot_count = 0
            for name, df in all_dfs.items():
                if "bits_down_total_round" in df.columns and "bits_up_total_round" in df.columns and col in df.columns:
                    df["bits_total"] = df["bits_down_total_round"] + df["bits_up_total_round"]
                    df["bits_cumsum"] = df["bits_total"].cumsum()
                    plt.plot(df["bits_cumsum"], df[col], marker="o", label=name)
                    plot_count += 1
            
            if plot_count > 0:
                plt.xlabel("Cumulative Bits")
                plt.ylabel(col)
                plt.grid(True)
                plt.legend()
                plt.title(f"{col} vs. Cumulative Bits Comparison")
                fig_path = out_dir / f"{col.replace('.', '_')}_vs_cum_bits_comparison.png"
                plt.savefig(fig_path, bbox_inches="tight")
                plt.close()
                print(f"[plot] 保存图像: {fig_path}")
            else:
                plt.close()
                print(f"[plot] 没有足够的 {col} 和通信比特数据可供绘制累计比特-精度图。")


# ==================== 命令行入口 ====================

class suppress:
    """简单的上下文管理器，用于忽略异常"""
    def __init__(self, exc_type=Exception):
        self.exc_type = exc_type
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, self.exc_type)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="单机 Fed-FLQ (server+multi-clients) 启动脚本")
    subparsers = parser.add_subparsers(dest="command")

    # train 子命令
    p_train = subparsers.add_parser("train", help="启动 server + 多 client 进行训练，并记录 CSV")
    p_train.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径，例如 configs/flq_config.yaml（默认使用 app.config 内部默认值）",
    )
    p_train.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="并行 client 数量，默认读取 Config.clients_per_round",
    )
    p_train.add_argument(
        "--csv",
        type=str,
        default="outputs/flq_local_stats.csv",
        help="监控结果 CSV 输出路径",
    )
    p_train.add_argument(
        "--no-monitor",
        action="store_true",
        help="不启动 /status 监控线程（则不会生成 CSV）",
    )

    # plot 子命令
    p_plot = subparsers.add_parser("plot", help="根据 CSV 绘制收敛与通信曲线")
    p_plot.add_argument("--csv", type=str, nargs='+', required=True, help="训练阶段生成的 CSV 文件路径 (可传入多个，用于对比)")
    p_plot.add_argument(
        "--out-dir",
        type=str,
        default="outputs/plots",
        help="图片输出目录",
    )

    args = parser.parse_args(argv)

    if args.command == "plot":
        csv_paths_dict = {Path(p).stem: Path(p) for p in args.csv}
        plot_flq(csv_paths_dict, Path(args.out_dir))
        return 0

    # 默认命令为 train
    if args.command in (None, "train"):
        csv_path = Path(args.csv)
        enable_monitor = not getattr(args, "no_monitor", False)
        return run_training(
            config_path=args.config,
            num_clients=args.num_clients,
            csv_path=csv_path,
            enable_monitor=enable_monitor,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
