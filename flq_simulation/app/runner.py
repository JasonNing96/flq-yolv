"""
FLQ-Fed 统一入口
简化的命令行接口
"""
import argparse
import subprocess
import time
import signal
import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def kill_existing_processes():
    """清理已有的 FLQ 进程（只清理 server 和 client）"""
    print("🧹 清理现有进程...")
    # 只清理 server 和 client，不要清理当前的 runner
    os.system("pkill -f 'app.server' 2>/dev/null")
    os.system("pkill -f 'app.client' 2>/dev/null")
    # 也清理旧版的进程
    os.system("pkill -f 'flq-fed.py' 2>/dev/null")
    time.sleep(1)
    print("✅ 清理完成\n")


def check_port_available(port: int = 8087) -> bool:
    """检查端口是否可用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


def start_server_mode(config_path: str = None):
    """启动服务器模式"""
    from .server import start_server
    start_server(config_path)


def start_client_mode(client_id: int, server_url: str = None, config_path: str = None):
    """启动客户端模式"""
    from .client import start_client
    start_client(client_id, server_url, config_path)


def train_full_mode(config_path: str = None, parallel: bool = False):
    """
    完整训练模式（自动启动服务器和客户端）
    
    Args:
        config_path: 配置文件路径
        parallel: 是否并行启动客户端（实验性）
    """
    print("="*70)
    print("🚀 FLQ-Fed 完整训练")
    print("="*70)
    
    # 1. 清理旧进程
    kill_existing_processes()
    
    # 2. 检查端口
    if not check_port_available(8087):
        print("❌ 端口 8087 已被占用，请先运行: pkill -f 'app.runner'")
        return 1
    
    # 3. 加载配置
    from .config import Config
    config = Config(config_path)
    print(f"\n📊 配置: {config}")
    print(f"👥 客户端数: {config.clients_per_round}")
    print(f"🌐 服务器: http://{config.server_host}:{config.server_port}")
    print("="*70 + "\n")
    
    processes = []
    log_files = []
    
    try:
        # 4. 启动服务器
        print("[1/2] 启动服务器...")
        server_cmd = [sys.executable, "-m", "app.server"]
        if config_path:
            server_cmd.extend(["--config", config_path])
        
        log_dir = PROJECT_ROOT / "outputs" / "server" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        server_log = open(log_dir / "server.log", "w", buffering=1)
        server_proc = subprocess.Popen(
            server_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append(("server", server_proc))
        log_files.append(server_log)
        
        print(f"✅ 服务器已启动 (PID: {server_proc.pid})")
        print(f"   日志: {log_dir}/server.log")
        
        # 等待服务器就绪
        print("⏳ 等待服务器就绪...")
        time.sleep(8)
        
        # 5. 启动客户端
        print(f"\n[2/2] 启动 {config.clients_per_round} 个客户端...")
        server_url = f"http://{config.server_host}:{config.server_port}"
        
        for i in range(1, config.clients_per_round + 1):
            client_cmd = [
                sys.executable, "-m", "app.client",
                str(i), server_url
            ]
            if config_path:
                client_cmd.extend(["--config", config_path])
            
            client_log_dir = PROJECT_ROOT / "outputs" / f"client{i}" / "logs"
            client_log_dir.mkdir(parents=True, exist_ok=True)
            
            client_log = open(client_log_dir / f"client{i}.log", "w", buffering=1)
            client_proc = subprocess.Popen(
                client_cmd,
                cwd=str(PROJECT_ROOT),
                stdout=client_log,
                stderr=subprocess.STDOUT,
                text=True
            )
            processes.append((f"client{i}", client_proc))
            log_files.append(client_log)
            
            print(f"✅ 客户端 #{i} 已启动 (PID: {client_proc.pid})")
            print(f"   日志: {client_log_dir}/client{i}.log")
            
            if not parallel:
                time.sleep(2)  # 串行启动，避免资源竞争
        
        print("\n" + "="*70)
        print("✅ 所有进程已启动")
        print("="*70)
        print("\n💡 提示:")
        print("  - 按 Ctrl+C 停止训练")
        print("  - 查看日志: tail -f outputs/server/logs/server.log")
        print("  - 监控状态: curl http://localhost:8087/status")
        print("  - 监控GPU: nvidia-smi")
        print("="*70 + "\n")
        
        # 6. 等待所有进程完成
        while True:
            time.sleep(5)
            
            # 检查服务器是否完成
            if server_proc.poll() is not None:
                print("\n🎉 训练完成！")
                break
            
            # 检查客户端是否异常退出
            for name, proc in processes[1:]:
                if proc.poll() is not None and proc.returncode != 0:
                    print(f"\n⚠️  {name} 异常退出 (code: {proc.returncode})")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号，正在停止所有进程...")
    
    finally:
        # 7. 清理进程和日志
        for name, proc in processes:
            if proc.poll() is None:
                print(f"🛑 停止 {name} (PID: {proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        
        for log_file in log_files:
            log_file.close()
        
        print("\n✅ 所有进程已停止")
        print("="*70)
    
    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FLQ-Fed 联邦学习框架 (简化版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启动服务器
  python -m app.runner server
  
  # 启动客户端
  python -m app.runner client --id 1
  
  # 完整训练（推荐）
  python -m app.runner train
  
  # 自定义配置
  python -m app.runner train --config configs/custom.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # server 命令
    server_parser = subparsers.add_parser("server", help="启动服务器")
    server_parser.add_argument("--config", type=str, help="配置文件路径")
    
    # client 命令
    client_parser = subparsers.add_parser("client", help="启动客户端")
    client_parser.add_argument("--id", type=int, required=True, help="客户端ID (1, 2, 3, ...)")
    client_parser.add_argument("--server", type=str, help="服务器地址 (默认: http://localhost:8087)")
    client_parser.add_argument("--config", type=str, help="配置文件路径")
    
    # train 命令
    train_parser = subparsers.add_parser("train", help="完整训练")
    train_parser.add_argument("--config", type=str, help="配置文件路径")
    train_parser.add_argument("--parallel", action="store_true", help="并行启动客户端（实验性）")
    
    args = parser.parse_args()
    
    if args.command == "server":
        start_server_mode(args.config)
    
    elif args.command == "client":
        start_client_mode(args.id, args.server, args.config)
    
    elif args.command == "train":
        return train_full_mode(args.config, args.parallel)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

