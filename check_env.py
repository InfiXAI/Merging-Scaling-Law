#!/usr/bin/env python3
"""
环境检查脚本 - 验证所有依赖是否正确安装
"""

import sys

def check_package(package_name, import_name=None):
    """检查包是否可以导入"""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:15s} {version}")
        return True
    except ImportError:
        print(f"❌ {package_name:15s} NOT INSTALLED")
        return False

def main():
    print("="*50)
    print("检查评估环境依赖")
    print("="*50)
    print()

    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
        ("swanlab", "swanlab"),
    ]

    all_installed = True
    missing_packages = []

    for display_name, import_name in packages:
        if not check_package(display_name, import_name):
            all_installed = False
            missing_packages.append(import_name)

    print()
    print("="*50)

    if all_installed:
        print("✅ 所有依赖已正确安装！")
        print()

        # 检查CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🚀 CUDA可用: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA版本: {torch.version.cuda}")
            else:
                print("ℹ️  CUDA不可用，将使用CPU")
        except:
            pass

        print()
        print("环境就绪，可以运行评估脚本！")
        return 0
    else:
        print("❌ 发现缺失的依赖包")
        print()
        print("请运行以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
        print()
        print("或使用自动安装脚本：")
        print("./auto_eval.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
