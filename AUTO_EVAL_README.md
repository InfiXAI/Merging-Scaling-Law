# 自动化模型评估脚本使用说明

## 功能特性

- ✅ 自动从GitHub下载评估代码
- ✅ 自动从HuggingFace下载模型
- ✅ 支持Conda虚拟环境（自动创建/激活）
- ✅ 自动安装依赖包
- ✅ 支持多种Conda安装方式（miniconda/anaconda/自定义路径）
- ✅ 支持pip镜像源加速（清华/阿里云/USTC等）
- ✅ 自动检测模型max_length并调整
- ✅ 支持CUDA/CPU设备自动选择
- ✅ 支持Callback API回调
- ✅ SwanLab日志记录
- ✅ 完整的进度输出和错误处理

## 快速开始

### 0. 检查环境（可选）
```bash
# 检查所有依赖是否已安装
python3 check_env.py
```

提供两种方式运行自动化评估：

### 方式1：使用Bash脚本（推荐）

```bash
# 1. 编辑 auto_eval.sh 修改配置
vi auto_eval.sh

# 2. 修改以下参数：
# CONDA_ENV_NAME="eval_env"          # Conda环境名称，留空则不使用conda
# MODEL_URL="your-huggingface-model"
# DATASET="algebra"  # 或 all
# MODEL_ID="your_model_name"
# BENCHMARK_ID="your_benchmark"
# PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"  # pip镜像源，留空使用默认

# 3. 运行脚本
./auto_eval.sh

# 国内用户加速技巧：
# - 使用清华镜像源: PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
# - 使用阿里云镜像: PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
# - 使用USTC镜像: PIP_INDEX_URL="https://mirrors.ustc.edu.cn/pypi/web/simple"
```

### 方式2：使用Python脚本

```bash
# 使用Conda环境（推荐）
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset algebra \
  --conda_env eval_env

# 使用系统Python（不使用Conda）
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset algebra \
  --conda_env ""

# 完整参数
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset algebra \
  --model_id my_model_v1 \
  --benchmark_id math_problems \
  --batch_size 8 \
  --max_length 2048 \
  --conda_env eval_env \
  --python_version 3.10
```

## 参数说明

### Conda环境参数（新增）

- `CONDA_ENV_NAME` / `--conda_env`: Conda环境名称
  - 设置环境名称将使用conda虚拟环境
  - 留空或设为""则使用系统Python
  - 默认: `eval_env`

- `CREATE_ENV_IF_NOT_EXISTS`: 环境不存在时是否自动创建（仅Bash脚本）
  - 默认: `true`

- `PYTHON_VERSION` / `--python_version`: 创建新环境时的Python版本
  - 默认: `3.10`

- `--no_create_env`: 如果conda环境不存在，不自动创建（仅Python脚本）

### 必须修改的参数

- `MODEL_URL` / `--model_url`: HuggingFace模型仓库地址
  - 例如: `microsoft/DialoGPT-medium`, `meta-llama/Llama-2-7b-hf`

- `MODEL_ID` / `--model_id`: 模型标识符
  - 例如: `my_model_v1`, `llama2_7b_finetuned`

### 可选参数

- `DATASET` / `--dataset`: 评估数据集
  - 选项: `algebra`, `analysis`, `discrete`, `geometry`, `number_theory`, `all`
  - 默认: `algebra`

- `BENCHMARK_ID` / `--benchmark_id`: 基准测试标识
  - 默认: `math_problems`

- `CALLBACK_URL` / `--callback_url`: 回调API地址
  - 默认: `http://147.8.92.70:22222/api/evaluate/callback`

- `BATCH_SIZE` / `--batch_size`: 批处理大小
  - 默认: `8`

- `MAX_LENGTH` / `--max_length`: 最大序列长度
  - 默认: `2048`

## 工作流程

1. **Conda环境设置**: 自动创建/激活conda虚拟环境（可选）
2. **下载代码**: 自动从GitHub克隆/更新评估代码
3. **下载模型**: 从HuggingFace自动下载指定模型
4. **安装依赖**: 在conda环境中自动安装所需Python包
5. **执行评估**: 调用eval.py进行模型评估
6. **发送回调**: 评估完成后自动发送结果到callback API

## 输出

- 评估结果保存在: `eval_workspace/Merging-EVAL/output/<task_id>/`
- 缓存文件保存在: `eval_workspace/Merging-EVAL/cache/<task_id>/`
- SwanLab可视化: 使用task_id查看实验结果

## 示例

### 示例1：使用Conda环境评估单个数据集（推荐）
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset algebra \
  --model_id dialogpt_test \
  --conda_env eval_env
```

### 示例2：评估所有数据集
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset all \
  --model_id dialogpt_full_eval \
  --conda_env eval_env
```

### 示例3：使用自定义Conda环境和参数
```bash
python3 auto_eval.py \
  --model_url meta-llama/Llama-2-7b-hf \
  --dataset geometry \
  --model_id llama2_geometry \
  --batch_size 16 \
  --max_length 4096 \
  --conda_env llama_eval \
  --python_version 3.11
```

### 示例4：不使用Conda（使用系统Python）
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset algebra \
  --model_id dialogpt_test \
  --conda_env ""
```

### 示例5：使用Bash脚本
```bash
# 编辑配置
vi auto_eval.sh
# 修改：
# CONDA_ENV_NAME="my_eval_env"
# MODEL_URL="microsoft/DialoGPT-medium"
# DATASET="algebra"

# 运行
./auto_eval.sh
```

## 注意事项

1. **首次使用建议使用Conda环境**，可以避免依赖冲突
2. 确保有足够的磁盘空间存储模型（通常5-20GB）
3. 首次运行需要下载模型，时间较长
4. 模型会缓存在本地，后续运行会更快
5. 日志输出已优化，仅显示关键信息
6. 需要联网访问HuggingFace和GitHub
7. Conda环境会自动创建（如果不存在）并安装依赖

## Conda环境管理

### 查看现有环境
```bash
conda env list
```

### 手动创建环境
```bash
conda create -n eval_env python=3.10 -y
conda activate eval_env
```

### 删除环境
```bash
conda deactivate
conda env remove -n eval_env
```

### 导出环境配置
```bash
conda activate eval_env
conda env export > environment.yml
```

## 故障排除

### Conda相关问题

**问题1：找不到conda命令**
```bash
# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh
# 或
source ~/anaconda3/etc/profile.d/conda.sh
```

**问题2：conda环境激活失败**
- 确保已安装Miniconda或Anaconda
- 运行 `conda init bash` 然后重启终端

**问题3：不想使用Conda**
```bash
# Bash脚本：设置 CONDA_ENV_NAME=""
# Python脚本：使用 --conda_env ""
```

### 模型下载失败
- 检查网络连接
- 确认HuggingFace模型地址正确
- 某些模型需要登录认证: `huggingface-cli login`

### 依赖安装失败
```bash
# 如果使用Conda
conda activate eval_env
pip install torch transformers datasets pandas tqdm requests swanlab

# 如果不使用Conda
pip3 install torch transformers datasets pandas tqdm requests swanlab
```

### 权限错误
```bash
chmod +x auto_eval.sh
chmod +x auto_eval.py
```

## 环境隔离优势

使用Conda虚拟环境的好处：
1. ✅ 依赖隔离，不影响系统Python环境
2. ✅ 可以同时维护多个Python版本
3. ✅ 方便分享和复现环境
4. ✅ 避免包版本冲突
5. ✅ 易于清理和重建
