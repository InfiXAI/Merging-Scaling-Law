# 自动化模型评估系统

## 📚 文件说明

### 核心脚本
- **[auto_eval.sh](auto_eval.sh)** - Bash自动评估脚本（推荐使用）
- **[auto_eval.py](auto_eval.py)** - Python自动评估脚本（更灵活）

### 文档
- **[AUTO_EVAL_README.md](AUTO_EVAL_README.md)** - 完整使用文档
- **[CONDA_QUICK_START.md](CONDA_QUICK_START.md)** - Conda快速开始指南
- **[DEPENDENCIES.md](DEPENDENCIES.md)** - 依赖包详细说明

### 工具脚本
- **[check_env.py](check_env.py)** - 环境检查脚本
- **[test_setup.sh](test_setup.sh)** - 安装测试脚本

### 依赖文件
- **[requirements-minimal.txt](requirements-minimal.txt)** - 最小依赖（推荐）
- **[requirements.txt](requirements.txt)** - 完整依赖（含训练/推理工具）
- **[ENV.txt](ENV.txt)** - 原始环境导出（参考用）

## 🚀 快速开始（3步）

### 步骤1：测试环境
```bash
./test_setup.sh
```

### 步骤2：配置参数
编辑 `auto_eval.sh`，修改以下配置：
```bash
CONDA_ENV_NAME="eval_env"              # Conda环境名
MODEL_URL="your-model-url"             # HuggingFace模型地址
MODEL_ID="your_model_id"               # 模型标识
DATASET="algebra"                      # 数据集
```

### 步骤3：运行评估
```bash
./auto_eval.sh
```

## ✅ 环境检查结果

运行 `./test_setup.sh` 后的检查结果：

```
✅ auto_eval.sh 存在
✅ auto_eval.py 存在
✅ Python已安装: Python 3.13.5
✅ Conda已安装: conda 25.7.0
✅ Git已安装
✅ torch 2.8.0
✅ transformers 4.56.0
✅ datasets 4.2.0
✅ pandas 2.3.2
✅ tqdm 4.67.1
✅ requests 2.32.4
✅ swanlab 0.7.2-dev
```

所有依赖已安装，环境就绪！

## 📋 核心功能

1. ✅ **自动下载代码** - 从GitHub自动克隆Merging-EVAL仓库
2. ✅ **自动下载模型** - 从HuggingFace自动下载指定模型
3. ✅ **Conda环境管理** - 自动创建/激活虚拟环境
4. ✅ **依赖自动安装** - 自动安装所有必需的Python包
5. ✅ **模型评估** - 调用eval.py执行评估
6. ✅ **Callback回调** - 自动发送结果到指定API
7. ✅ **简洁日志** - 优化的日志输出

## 🎯 使用示例

### 示例1：使用Bash脚本（最简单）
```bash
# 1. 编辑配置
vi auto_eval.sh

# 2. 运行
./auto_eval.sh
```

### 示例2：使用Python脚本
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset algebra \
  --model_id dialogpt_test \
  --conda_env eval_env
```

### 示例3：评估所有数据集
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset all \
  --model_id dialogpt_full \
  --conda_env eval_env
```

### 示例4：不使用Conda
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset algebra \
  --model_id dialogpt_test \
  --conda_env ""
```

## 🔧 配置参数

### 必须修改的参数
- `MODEL_URL` - HuggingFace模型仓库地址
- `MODEL_ID` - 模型标识符

### 可选参数
- `CONDA_ENV_NAME` - Conda环境名称（默认：eval_env）
- `DATASET` - 评估数据集（默认：algebra，可选：all）
- `BATCH_SIZE` - 批处理大小（默认：8）
- `MAX_LENGTH` - 最大序列长度（默认：2048）
- `CALLBACK_URL` - 回调API地址
- `BENCHMARK_ID` - 基准测试标识

## 📊 工作流程

```
1. 检测/创建Conda环境 (eval_env)
   ↓
2. 克隆GitHub代码仓库
   ↓
3. 下载HuggingFace模型
   ↓
4. 安装Python依赖包
   ↓
5. 执行模型评估
   ↓
6. 发送Callback回调
```

## 📁 输出位置

- 评估结果：`eval_workspace/Merging-EVAL/output/<task_id>/`
- 缓存文件：`eval_workspace/Merging-EVAL/cache/<task_id>/`
- SwanLab实验：使用task_id查看

## 🛠️ 故障排除

### 问题1：缺少依赖包
```bash
# 运行环境检查
python3 check_env.py

# 手动安装
pip install -r requirements.txt
```

### 问题2：Conda未初始化
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

### 问题3：模型下载失败
```bash
# 需要HuggingFace认证
huggingface-cli login
```

## 📚 详细文档

- 完整使用说明：[AUTO_EVAL_README.md](AUTO_EVAL_README.md)
- Conda快速指南：[CONDA_QUICK_START.md](CONDA_QUICK_START.md)
- 依赖包说明：[DEPENDENCIES.md](DEPENDENCIES.md)

## 💡 最佳实践

1. **首次使用建议**：使用Conda环境，避免依赖冲突
2. **测试运行**：先用单个数据集（algebra）测试
3. **生产运行**：使用所有数据集（all）进行完整评估
4. **环境隔离**：为不同项目创建不同的conda环境
5. **定期清理**：删除不需要的缓存和旧环境

## 🎉 开始使用

```bash
# 一键测试环境
./test_setup.sh

# 如果通过，直接运行
./auto_eval.sh
```

祝评估顺利！🚀
