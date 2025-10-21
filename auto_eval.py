#!/usr/bin/env python3
"""
自动化模型评估脚本
自动从GitHub下载代码，从HuggingFace下载模型，并执行评估
支持Conda虚拟环境
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# ========================================
# 默认配置
# ========================================
DEFAULT_CONFIG = {
    "conda_env": "eval_env",  # Conda环境名称，设为None则不使用conda
    "python_version": "3.10",  # 创建新环境时使用的Python版本
    "create_env": True,  # 如果环境不存在是否自动创建
    "model_url": "microsoft/DialoGPT-medium",
    "dataset": "algebra",
    "callback_url": "http://147.8.92.70:22222/api/evaluate/callback",
    "model_id": "my_model_v1",
    "benchmark_id": "math_problems",
    "batch_size": 8,
    "max_length": 2048,
    "repo_url": "https://github.com/maxuan1798/Merging-EVAL.git",
    "work_dir": "./eval_workspace"
}

def run_command(cmd, silent=False, shell=False):
    """执行命令"""
    try:
        if silent:
            subprocess.run(cmd, check=True, capture_output=True, text=True, shell=shell)
        else:
            subprocess.run(cmd, check=True, shell=shell)
        return True
    except subprocess.CalledProcessError as e:
        if not silent:
            print(f"Command failed: {cmd if shell else ' '.join(cmd)}")
            if hasattr(e, 'stderr') and e.stderr:
                print(e.stderr)
        return False

def check_conda_env(env_name):
    """检查conda环境是否存在"""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        return env_name in result.stdout
    except:
        return False

def setup_conda_env(config):
    """设置Conda环境"""
    env_name = config.get("conda_env")

    if not env_name:
        print("No Conda environment specified, using system Python")
        return sys.executable

    print(f"Setting up Conda environment: {env_name}")

    # 检查conda是否安装
    if not run_command(["conda", "--version"], silent=True):
        print("Error: Conda not found. Please install Conda or set conda_env to None.")
        sys.exit(1)

    # 检查环境是否存在
    if check_conda_env(env_name):
        print(f"Using existing environment: {env_name}")
    else:
        if config.get("create_env", True):
            print(f"Creating new Conda environment: {env_name} with Python {config['python_version']}")
            if not run_command([
                "conda", "create", "-n", env_name,
                f"python={config['python_version']}", "-y"
            ]):
                print("Failed to create Conda environment")
                sys.exit(1)
        else:
            print(f"Error: Conda environment '{env_name}' not found.")
            sys.exit(1)

    # 获取conda环境的Python路径
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "which", "python"],
            capture_output=True,
            text=True,
            check=True
        )
        python_path = result.stdout.strip()
        print(f"Using Conda Python: {python_path}")
        return python_path
    except:
        print(f"Error: Failed to get Python path from environment '{env_name}'")
        sys.exit(1)

def setup_environment(config):
    """设置环境"""
    work_dir = config["work_dir"]
    repo_dir = os.path.join(work_dir, "Merging-EVAL")

    # 创建工作目录
    os.makedirs(work_dir, exist_ok=True)

    # 克隆或更新代码仓库
    if os.path.exists(repo_dir):
        print("Updating repository...")
        os.chdir(repo_dir)
        run_command(["git", "pull", "-q", "origin", "main"], silent=True)
        os.chdir("../..")
    else:
        print("Cloning repository...")
        os.chdir(work_dir)
        run_command(["git", "clone", "-q", config["repo_url"]], silent=True)
        os.chdir("..")

    return repo_dir

def install_dependencies(python_cmd, conda_env=None):
    """安装依赖"""
    print("Checking dependencies...")
    packages = ["torch", "transformers", "datasets", "pandas", "tqdm", "requests", "swanlab"]

    for package in packages:
        # 检查包是否已安装
        check_cmd = [python_cmd, "-c", f"import {package}"]
        if conda_env:
            check_cmd = ["conda", "run", "-n", conda_env] + check_cmd

        result = subprocess.run(check_cmd, capture_output=True)
        if result.returncode != 0:
            print(f"Installing {package}...")
            install_cmd = [python_cmd, "-m", "pip", "install", "-q", package]
            if conda_env:
                install_cmd = ["conda", "run", "-n", conda_env] + install_cmd
            run_command(install_cmd, silent=True)

def run_evaluation(repo_dir, config, python_cmd):
    """运行评估"""
    task_id = f"eval_task_{int(datetime.now().timestamp())}"

    eval_script = os.path.join(repo_dir, "scripts", "eval.py")
    data_dir = os.path.join(repo_dir, "data", "eval_partial")
    output_dir = os.path.join(repo_dir, "output", task_id)
    cache_dir = os.path.join(repo_dir, "cache", task_id)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Model: {config['model_url']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Task ID: {task_id}")
    print(f"{'='*50}\n")

    # 构建评估命令
    cmd = [
        python_cmd, eval_script,
        "--model", config["model_url"],
        "--tokenizer", config["model_url"],
        "--output", output_dir,
        "--cache_dir", cache_dir,
        "--experiment_name", task_id,
        "--callback_url", config["callback_url"],
        "--task_id", task_id,
        "--model_id", config["model_id"],
        "--benchmark_id", config["benchmark_id"],
        "--max_length", str(config["max_length"]),
        "--batch_size", str(config["batch_size"]),
        "--use_swanlab"
    ]

    # 如果使用conda环境，需要通过conda run执行
    if config.get("conda_env"):
        cmd = ["conda", "run", "-n", config["conda_env"]] + cmd

    # 根据数据集类型选择参数
    if config["dataset"] == "all":
        cmd.extend(["--dataset", data_dir])
    else:
        dataset_file = os.path.join(data_dir, f"{config['dataset']}.json")
        cmd.extend(["--file", dataset_file])

    # 执行评估
    print("Starting evaluation...\n")
    if run_command(cmd):
        print(f"\n{'='*50}")
        print("Evaluation completed!")
        print(f"Results: {output_dir}")
        print(f"{'='*50}")
        return True
    else:
        print("\nEvaluation failed!")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="自动化模型评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model_url", default=DEFAULT_CONFIG["model_url"],
                       help="HuggingFace模型地址")
    parser.add_argument("--dataset", default=DEFAULT_CONFIG["dataset"],
                       choices=["algebra", "analysis", "discrete", "geometry", "number_theory", "all"],
                       help="评估数据集")
    parser.add_argument("--callback_url", default=DEFAULT_CONFIG["callback_url"],
                       help="回调API地址")
    parser.add_argument("--model_id", default=DEFAULT_CONFIG["model_id"],
                       help="模型标识")
    parser.add_argument("--benchmark_id", default=DEFAULT_CONFIG["benchmark_id"],
                       help="基准测试标识")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                       help="批处理大小")
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"],
                       help="最大序列长度")
    parser.add_argument("--conda_env", default=DEFAULT_CONFIG["conda_env"],
                       help="Conda环境名称（留空则使用系统Python）")
    parser.add_argument("--python_version", default=DEFAULT_CONFIG["python_version"],
                       help="创建新环境时使用的Python版本")
    parser.add_argument("--no_create_env", action="store_true",
                       help="如果conda环境不存在，不自动创建")

    args = parser.parse_args()

    # 准备配置
    config = {
        "model_url": args.model_url,
        "dataset": args.dataset,
        "callback_url": args.callback_url,
        "model_id": args.model_id,
        "benchmark_id": args.benchmark_id,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "conda_env": args.conda_env if args.conda_env else None,
        "python_version": args.python_version,
        "create_env": not args.no_create_env,
        "repo_url": DEFAULT_CONFIG["repo_url"],
        "work_dir": DEFAULT_CONFIG["work_dir"]
    }

    # 设置conda环境（如果需要）
    python_cmd = setup_conda_env(config)

    # 设置环境
    repo_dir = setup_environment(config)

    # 安装依赖
    install_dependencies(python_cmd, config.get("conda_env"))

    # 运行评估
    success = run_evaluation(repo_dir, config, python_cmd)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
