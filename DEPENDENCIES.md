# 依赖包列表

## 📋 依赖文件说明

本项目提供三个依赖文件：

1. **requirements-minimal.txt** - 最小依赖（推荐用于评估）
   - 仅包含auto_eval脚本必需的包
   - 适合快速安装和基础评估

2. **requirements.txt** - 完整依赖（合并了ENV.txt）
   - 包含所有训练、推理、优化工具
   - 适合完整的深度学习环境

3. **ENV.txt** - 原始完整环境导出
   - 保留用于参考

## 基础依赖（必需）

自动评估脚本需要以下Python包：

### 核心依赖
1. **torch** - PyTorch深度学习框架
2. **transformers** - HuggingFace Transformers库
3. **datasets** - HuggingFace Datasets库（用于加载数据集）

### 数据处理
4. **pandas** - 数据分析和处理
5. **tqdm** - 进度条显示

### 网络和监控
6. **requests** - HTTP请求（用于callback）
7. **swanlab** - 实验跟踪和可视化

## 扩展依赖（可选）

完整的requirements.txt还包括：

### 训练加速
- **accelerate** - 分布式训练
- **bitsandbytes** - 量化训练
- **peft** - 参数高效微调
- **deepspeed** - 大规模训练

### 推理优化
- **vllm** - 高效推理引擎
- **xformers** - 优化的Transformer实现
- **triton** - GPU编程

### 监控工具
- **wandb** - 实验跟踪
- **nvitop** - GPU监控
- **prometheus** - 指标监控

## 自动安装

脚本会自动安装所有依赖：

### 使用Bash脚本
```bash
./auto_eval.sh
# 自动安装到conda环境或系统Python
```

### 使用Python脚本
```bash
python3 auto_eval.py --conda_env eval_env
# 自动安装到指定conda环境
```

## 手动安装

### 使用Conda环境（推荐）
```bash
# 创建环境
conda create -n eval_env python=3.10 -y
conda activate eval_env

# 安装依赖
pip install torch transformers datasets pandas tqdm requests swanlab
```

### 使用系统Python
```bash
pip3 install torch transformers datasets pandas tqdm requests swanlab
```

### 使用requirements文件

#### 方式1：最小依赖（推荐）
```bash
# 仅安装评估必需的包
pip install -r requirements-minimal.txt
```

#### 方式2：完整依赖
```bash
# 安装所有包（包含训练、推理等）
pip install -r requirements.txt
```

#### 方式3：分步安装
```bash
# 先安装基础包
pip install torch transformers datasets pandas tqdm requests swanlab

# 可选：安装加速包
pip install accelerate bitsandbytes peft

# 可选：安装推理优化
pip install vllm xformers
```

## 版本要求

- Python: 3.8 - 3.11（推荐3.10）
- torch: >= 2.0.0
- transformers: >= 4.30.0
- datasets: >= 2.12.0
- pandas: >= 2.0.0
- tqdm: >= 4.65.0
- requests: >= 2.31.0
- swanlab: >= 0.3.0

## 常见问题

### torch安装失败
```bash
# CPU版本
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8版本
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1版本
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### transformers安装慢
```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers
```

### datasets包导入错误
```bash
# 确保安装的是 datasets 而不是 dataset
pip uninstall dataset datasets -y
pip install datasets
```

## 验证安装

### 检查所有包
```bash
python3 << EOF
import torch
import transformers
import datasets
import pandas
import tqdm
import requests
import swanlab

print("✅ torch:", torch.__version__)
print("✅ transformers:", transformers.__version__)
print("✅ datasets:", datasets.__version__)
print("✅ pandas:", pandas.__version__)
print("✅ tqdm:", tqdm.__version__)
print("✅ requests:", requests.__version__)
print("✅ swanlab:", swanlab.__version__)
print("\n所有依赖已正确安装！")
EOF
```

### 检查CUDA可用性
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 故障排除

### ModuleNotFoundError: No module named 'datasets'
```bash
pip install datasets
```

### ImportError: No module named 'swanlab'
```bash
pip install swanlab
```

### 权限错误
```bash
# 使用用户安装
pip install --user torch transformers datasets pandas tqdm requests swanlab
```

### Conda环境问题
```bash
# 确保在正确的环境中
conda activate eval_env
which python
pip list | grep -E "(torch|transformers|datasets)"
```
