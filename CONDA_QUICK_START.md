# Conda虚拟环境快速开始指南

## 🚀 最快开始方式

### 使用Bash脚本（一键运行）
```bash
# 1. 编辑配置（只需修改这3项）
vi auto_eval.sh

# 修改：
CONDA_ENV_NAME="eval_env"           # 环境名
MODEL_URL="your-model-url"          # 模型地址
MODEL_ID="your_model_id"            # 模型ID

# 2. 运行（自动完成一切）
./auto_eval.sh
```

### 使用Python脚本
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --model_id my_model \
  --dataset algebra \
  --conda_env eval_env
```

## 📋 配置模板

### Bash脚本配置模板
```bash
# ========================================
# Conda环境配置
# ========================================
CONDA_ENV_NAME="eval_env"              # 🔧 修改：环境名称
CREATE_ENV_IF_NOT_EXISTS=true          # ✅ 自动创建环境
PYTHON_VERSION="3.10"                  # 🐍 Python版本

# ========================================
# 模型配置
# ========================================
MODEL_URL="microsoft/DialoGPT-medium"  # 🔧 修改：HF模型地址
DATASET="algebra"                      # 📊 数据集选择
MODEL_ID="my_model_v1"                 # 🔧 修改：模型标识
BENCHMARK_ID="math_problems"           # 📈 基准测试ID

# ========================================
# Callback配置
# ========================================
CALLBACK_URL="http://147.8.92.70:22222/api/evaluate/callback"
TASK_ID="eval_task_$(date +%s)"        # 自动生成

# ========================================
# 评估参数
# ========================================
BATCH_SIZE=8
MAX_LENGTH=2048
```

### Python脚本常用命令
```bash
# 基础用法
python3 auto_eval.py \
  --model_url <模型地址> \
  --model_id <模型ID> \
  --conda_env <环境名>

# 完整参数
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --model_id my_model \
  --dataset algebra \
  --conda_env eval_env \
  --python_version 3.10 \
  --batch_size 8 \
  --max_length 2048 \
  --callback_url http://147.8.92.70:22222/api/evaluate/callback \
  --benchmark_id math_problems
```

## 🔄 工作流程

脚本会自动完成：
1. ✅ 创建/激活Conda环境（名称：eval_env，Python 3.10）
2. ✅ 克隆GitHub代码仓库
3. ✅ 下载HuggingFace模型
4. ✅ 安装依赖包（torch, transformers等）
5. ✅ 执行模型评估
6. ✅ 发送Callback结果

## 💡 常见场景

### 场景1：首次使用（推荐）
```bash
# 使用默认配置，自动创建环境
./auto_eval.sh
```

### 场景2：使用已有Conda环境
```bash
# 设置 CONDA_ENV_NAME 为已存在的环境名
CONDA_ENV_NAME="my_existing_env"
./auto_eval.sh
```

### 场景3：不使用Conda
```bash
# 设置为空字符串
CONDA_ENV_NAME=""
./auto_eval.sh
```

### 场景4：评估多个数据集
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --model_id my_model \
  --dataset all \
  --conda_env eval_env
```

### 场景5：自定义Python版本
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --model_id my_model \
  --conda_env my_env \
  --python_version 3.11
```

## ⚙️ Conda环境管理

### 查看所有环境
```bash
conda env list
```

### 激活环境
```bash
conda activate eval_env
```

### 退出环境
```bash
conda deactivate
```

### 删除环境
```bash
conda env remove -n eval_env
```

### 查看环境中的包
```bash
conda activate eval_env
pip list
```

## 🐛 常见问题

### Q1: 找不到conda命令
```bash
# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh
# 或
source ~/anaconda3/etc/profile.d/conda.sh

# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
```

### Q2: 环境创建失败
```bash
# 手动创建
conda create -n eval_env python=3.10 -y
conda activate eval_env

# 然后运行脚本
./auto_eval.sh
```

### Q3: 包安装失败
```bash
# 手动安装
conda activate eval_env
pip install torch transformers datasets pandas tqdm requests swanlab
```

### Q4: 想要完全控制
```bash
# 禁用自动创建环境
CREATE_ENV_IF_NOT_EXISTS=false  # Bash脚本

# 或
python3 auto_eval.py --no_create_env  # Python脚本
```

## 📊 参数优先级

| 参数 | Bash脚本变量 | Python参数 | 默认值 |
|------|------------|-----------|--------|
| Conda环境 | CONDA_ENV_NAME | --conda_env | eval_env |
| Python版本 | PYTHON_VERSION | --python_version | 3.10 |
| 模型地址 | MODEL_URL | --model_url | microsoft/DialoGPT-medium |
| 数据集 | DATASET | --dataset | algebra |
| 批大小 | BATCH_SIZE | --batch_size | 8 |
| 最大长度 | MAX_LENGTH | --max_length | 2048 |

## 📝 注意事项

1. **首次运行时间较长**：需要创建环境、下载模型
2. **后续运行快速**：环境和模型已缓存
3. **环境隔离**：不会影响系统Python
4. **自动化程度高**：最小化手动操作
5. **灵活配置**：可根据需求调整所有参数

## 🎯 推荐配置

### 开发环境
```bash
CONDA_ENV_NAME="eval_dev"
PYTHON_VERSION="3.10"
BATCH_SIZE=4
MAX_LENGTH=1024
```

### 生产环境
```bash
CONDA_ENV_NAME="eval_prod"
PYTHON_VERSION="3.10"
BATCH_SIZE=16
MAX_LENGTH=2048
```

### 测试环境
```bash
CONDA_ENV_NAME="eval_test"
PYTHON_VERSION="3.10"
DATASET="algebra"  # 单个数据集快速测试
```
