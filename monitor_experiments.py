#!/usr/bin/env python3
"""
监控消融实验进度
"""

import time
import subprocess
from pathlib import Path

def main():
    print("开始监控消融实验进度...")
    print("按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            print("\n" + "="*80)
            print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # 运行汇总脚本
            result = subprocess.run(
                ["python", "summarize_ablation_results.py"],
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            
            # 检查是否所有实验都完成
            if result.returncode == 0:
                print("\n✓ 所有实验已完成！")
                break
            
            # 等待60秒后再次检查
            print("\n下次检查时间: " + time.strftime('%H:%M:%S', time.localtime(time.time() + 60)))
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    main()
