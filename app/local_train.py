"""
FLQ-Fed 单节点训练脚本
在本地同时运行 server 和 client，适合快速测试
"""
import threading
import multiprocessing
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_server_thread(config_path: str = None):
    """在后台线程运行服务器"""
    from app.server import start_server
    try:
        start_server(config_path)
    except KeyboardInterrupt:
        pass


def run_client_process(client_id: int, server_url: str, config_path: str = None):
    """在独立进程中运行客户端"""
    from app.client import start_client
    try:
        start_client(client_id, server_url, config_path)
    except KeyboardInterrupt:
        pass


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FLQ-Fed 单节点训练（快速测试）")
    parser.add_argument("--config", type=str, default="configs/flq_config.yaml", help="配置文件路径")
    parser.add_argument("--clients", type=int, default=1, help="客户端数量")
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 FLQ-Fed 单节点训练")
    print("="*70)
    
    # 加载配置
    from app.config import Config
    config = Config(args.config)
    
    print(f"\n📊 配置: {config}")
    print(f"👥 客户端数: {args.clients}")
    print(f"🔄 训练轮数: {config.rounds}")
    print(f"📦 本地 Epoch: {config.local_epochs}")
    print("="*70 + "\n")
    
    server_url = f"http://{config.server_host}:{config.server_port}"
    
    try:
        # 1. 在后台线程启动服务器
        print("[1/2] 启动服务器（后台线程）...")
        server_thread = threading.Thread(
            target=run_server_thread,
            args=(args.config,),
            daemon=True
        )
        server_thread.start()
        
        # 等待服务器就绪
        print("⏳ 等待服务器启动...")
        time.sleep(10)
        print("✅ 服务器已就绪\n")
        
        # 2. 并行启动所有客户端（在独立进程中 - 真正的并行）
        print(f"[2/2] 并行启动 {args.clients} 个客户端...\n")

        client_processes = []
        for i in range(1, args.clients + 1):
            print(f"🚀 启动客户端 #{i} (独立进程)")
            process = multiprocessing.Process(
                target=run_client_process,
                args=(i, server_url, args.config)
            )
            process.start()
            client_processes.append(process)
            time.sleep(1)  # 稍微错开启动时间

        print(f"\n✅ 所有 {args.clients} 个客户端已启动，等待训练完成...\n")

        # 等待所有客户端完成
        for i, process in enumerate(client_processes, 1):
            process.join()
            print(f"✅ 客户端 #{i} 已完成")
        
        print("\n" + "="*70)
        print("🎉 所有客户端训练完成！")
        print("="*70)
        print("\n📁 查看结果:")
        print("   - 全局模型: outputs/server/checkpoints/")
        print("   - 客户端结果: outputs/client*/runs/")
        print("="*70)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号，停止训练...")
    
    return 0


if __name__ == "__main__":
    # 对于 multiprocessing，需要设置启动方法
    multiprocessing.set_start_method('spawn', force=True)
    sys.exit(main())

