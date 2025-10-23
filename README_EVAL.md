# 自动化模型评估系统

## 📚 文件说明

### 核心脚本
- **[auto_eval.sh](auto_eval.sh)** - Bash自动评估脚本（推荐使用）
- **[auto_eval.py](auto_eval.py)** - Python自动评估脚本（更灵活）
- **[scripts/eval.py](scripts/eval.py)** - 统一评估脚本（支持在线/离线模式）
- **[src/merge/main_merging.py](src/merge/main_merging.py)** - 模型合并脚本（支持在线/离线模式）

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

## 🔧 直接使用评估脚本

### 使用统一评估脚本 (scripts/eval.py)

#### 在线模式（默认）
```bash
# 评估单个数据集
python3 scripts/eval.py \
  --model /path/to/model \
  --tokenizer /path/to/tokenizer \
  --dataset /path/to/data \
  --output /path/to/output

# 评估特定文件
python3 scripts/eval.py \
  --model /path/to/model \
  --tokenizer /path/to/tokenizer \
  --file /path/to/data/algebra.json \
  --output /path/to/output

# 评估数据切片
python3 scripts/eval.py \
  --model /path/to/model \
  --tokenizer /path/to/tokenizer \
  --file /path/to/data/algebra.json \
  --indices "1-10,15,20-22" \
  --output /path/to/output
```

#### 离线模式
```bash
# 离线评估（使用本地文件）
python3 scripts/eval.py \
  --model /path/to/model \
  --tokenizer /path/to/tokenizer \
  --file /path/to/data/algebra.json \
  --output /path/to/output \
  --offline

# 离线评估并自定义输出目录
python3 scripts/eval.py \
  --model /path/to/model \
  --tokenizer /path/to/tokenizer \
  --file /path/to/data/algebra.json \
  --output /path/to/output \
  --offline \
  --run_name "my_experiment"
```

#### 高级功能
```bash
# 启用SwanLab日志记录
python3 scripts/eval.py \
  --model /path/to/model \
  --tokenizer /path/to/tokenizer \
  --dataset /path/to/data \
  --output /path/to/output \
  --use_swanlab \
  --experiment_name "my_experiment"

# 启用回调机制
python3 scripts/eval.py \
  --model /path/to/model \
  --tokenizer /path/to/tokenizer \
  --dataset /path/to/data \
  --output /path/to/output \
  --callback_url "https://api.example.com/callback" \
  --task_id "task_123" \
  --model_id "my_model" \
  --benchmark_id "benchmark_1"
```

## 🔗 模型合并功能

### 使用模型合并脚本 (src/merge/main_merging.py)

#### 在线模式（默认）
```bash
# 基本模型合并
python3 src/merge/main_merging.py \
  --merge_method average_merging \
  --base_model /path/to/base/model \
  --models_to_merge /path/to/model1,/path/to/model2 \
  --output_dir /path/to/output

# 高级合并配置
python3 src/merge/main_merging.py \
  --merge_method task_arithmetic \
  --base_model /path/to/base/model \
  --models_to_merge /path/to/model1,/path/to/model2,/path/to/model3 \
  --output_dir /path/to/output \
  --scaling_coefficient 0.5 \
  --param_value_mask_rate 0.8 \
  --use_gpu
```

#### 离线模式
```bash
# 离线模型合并（使用本地文件）
python3 src/merge/main_merging.py \
  --merge_method average_merging \
  --base_model /path/to/base/model \
  --models_to_merge /path/to/model1,/path/to/model2 \
  --output_dir /path/to/output \
  --offline

# 离线合并并排除特定参数
python3 src/merge/main_merging.py \
  --merge_method task_arithmetic \
  --base_model /path/to/base/model \
  --models_to_merge /path/to/model1,/path/to/model2 \
  --output_dir /path/to/output \
  --offline \
  --exclude_param_names_regex "lm_head,embed_tokens" \
  --scaling_coefficient 0.3
```

#### 支持的合并方法
- `average_merging` - 平均合并（默认）
- `task_arithmetic` - 任务算术合并
- `dare_ties` - DARE-TIES合并
- `ties` - TIES合并
- `magnitude_prune` - 幅度剪枝合并

## 🔧 配置参数

### 自动评估脚本参数

#### 必须修改的参数
- `MODEL_URL` - HuggingFace模型仓库地址
- `MODEL_ID` - 模型标识符

#### 可选参数
- `CONDA_ENV_NAME` - Conda环境名称（默认：eval_env）
- `DATASET` - 评估数据集（默认：algebra，可选：all）
- `BATCH_SIZE` - 批处理大小（默认：8）
- `MAX_LENGTH` - 最大序列长度（默认：2048）
- `CALLBACK_URL` - 回调API地址
- `BENCHMARK_ID` - 基准测试标识

### 直接评估脚本参数 (scripts/eval.py)

#### 必需参数
- `--model` - 模型路径
- `--tokenizer` - Tokenizer路径
- `--dataset` 或 `--file` - 数据集路径或文件路径

#### 可选参数
- `--output` - 输出目录（默认：./output）
- `--batch_size` - 批处理大小（默认：10）
- `--max_length` - 最大序列长度（默认：2048）
- `--indices` - 数据切片索引（如："1-10,15,20-22"）
- `--offline` - 离线模式（使用本地文件）
- `--run_name` - 自定义输出目录名称
- `--no_cache` - 禁用缓存
- `--cache_dir` - 缓存目录（默认：./cache）
- `--use_swanlab` - 启用SwanLab日志记录
- `--experiment_name` - 实验名称
- `--callback_url` - 回调API地址
- `--task_id` - 任务ID
- `--model_id` - 模型ID
- `--benchmark_id` - 基准测试ID

### 模型合并脚本参数 (src/merge/main_merging.py)

#### 必需参数
- `--merge_method` - 合并方法
- `--base_model` - 基础模型路径
- `--models_to_merge` - 要合并的模型路径（逗号分隔）
- `--output_dir` - 输出目录

#### 可选参数
- `--offline` - 离线模式（使用本地文件）
- `--scaling_coefficient` - 缩放系数（默认：1.0）
- `--param_value_mask_rate` - 参数值掩码率（默认：0.8）
- `--use_gpu` - 使用GPU
- `--mask_apply_method` - 掩码应用方法（默认：average_merging）
- `--weight_mask_rates` - 权重掩码率（逗号分隔）
- `--exclude_param_names_regex` - 排除的参数名称正则表达式

## 📊 工作流程

### 自动评估工作流程
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

### 直接评估工作流程
```
1. 加载模型和Tokenizer
   ↓
2. 处理数据集（支持切片）
   ↓
3. 执行评估计算
   ↓
4. 保存结果和日志
   ↓
5. 发送回调（可选）
```

### 模型合并工作流程
```
1. 加载基础模型
   ↓
2. 加载候选模型
   ↓
3. 执行合并算法
   ↓
4. 保存合并后的模型
   ↓
5. 保存Tokenizer
```

## 📁 输出位置

### 自动评估输出
- 评估结果：`eval_workspace/Merging-EVAL/output/<task_id>/`
- 缓存文件：`eval_workspace/Merging-EVAL/cache/<task_id>/`
- SwanLab实验：使用task_id查看

### 直接评估输出
- 评估结果：`<output_dir>/<model_name>/<domain>/results.csv`
- 缓存文件：`<cache_dir>/<cache_key>.pkl`
- SwanLab实验：使用experiment_name查看

### 模型合并输出
- 合并模型：`<output_dir>/pytorch_model.bin`
- 配置文件：`<output_dir>/config.json`
- Tokenizer：`<output_dir>/tokenizer.json`

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

### 评估最佳实践
1. **首次使用建议**：使用Conda环境，避免依赖冲突
2. **测试运行**：先用单个数据集（algebra）测试
3. **生产运行**：使用所有数据集（all）进行完整评估
4. **环境隔离**：为不同项目创建不同的conda环境
5. **定期清理**：删除不需要的缓存和旧环境
6. **离线模式**：在无网络环境中使用`--offline`参数
7. **数据切片**：使用`--indices`参数测试特定数据子集
8. **自定义输出**：使用`--run_name`参数自定义输出目录名称

### 模型合并最佳实践
1. **模型兼容性**：确保所有模型具有相同的架构
2. **内存管理**：大模型合并时注意GPU内存使用
3. **参数排除**：使用`--exclude_param_names_regex`排除不兼容的参数
4. **缩放系数**：根据任务调整`--scaling_coefficient`参数
5. **离线合并**：在无网络环境中使用`--offline`参数
6. **备份原模型**：合并前备份原始模型文件

## 🎉 开始使用

### 快速开始
```bash
# 一键测试环境
./test_setup.sh

# 如果通过，直接运行
./auto_eval.sh
```

### 实际使用示例

#### 评估合并后的模型
```bash
# 设置环境变量
export TRANSFORMERS_NO_TORCHVISION=1 \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
       CUDA_VISIBLE_DEVICES=2

# 评估单个数据集（离线模式）
python3 scripts/eval.py \
  --model /path/to/merged/model \
  --tokenizer /path/to/merged/model \
  --file /path/to/data/algebra.json \
  --output /path/to/output \
  --batch_size 1 \
  --max_length 2048 \
  --offline

# 评估所有数据集
python3 scripts/eval.py \
  --model /path/to/merged/model \
  --tokenizer /path/to/merged/model \
  --dataset /path/to/data \
  --output /path/to/output \
  --batch_size 1 \
  --max_length 2048 \
  --offline
```

#### 合并多个模型
```bash
# 合并多个模型
python3 src/merge/main_merging.py \
  --merge_method task_arithmetic \
  --base_model /path/to/base/model \
  --models_to_merge /path/to/model1,/path/to/model2,/path/to/model3 \
  --output_dir /path/to/merged/model \
  --scaling_coefficient 0.1 \
  --offline \
  --use_gpu
```

#### 完整工作流程
```bash
# 1. 合并模型
python3 src/merge/main_merging.py \
  --merge_method task_arithmetic \
  --base_model /path/to/base/model \
  --models_to_merge /path/to/model1,/path/to/model2 \
  --output_dir /path/to/merged/model \
  --scaling_coefficient 0.1 \
  --offline

# 2. 评估合并后的模型
python3 scripts/eval.py \
  --model /path/to/merged/model \
  --tokenizer /path/to/merged/model \
  --dataset /path/to/data \
  --output /path/to/eval_results \
  --offline \
  --run_name "merged_model_eval"
```

示例命令行
```bash
export TRANSFORMERS_NO_TORCHVISION=1 
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 && 
python3 /zju_0038/co-genai/Merging-Scaling-Law-main/scripts/evalv2_clean.py 
--model /zju_0038/jinjia/workspace/Merging-Scaling-Law-main/models/merged/
Llama-3B-cmb/task_arithmetic_9/sc0.1_r0/6p3h --tokenizer /zju_0038/jinjia/
workspace/Merging-Scaling-Law-main/models/merged/Llama-3B-cmb/
task_arithmetic_9/sc0.1_r0/6p3h --file /zju_0038/jinjia/workspace/
Merging-Scaling-Law-main/data/eval_partial/algebra.json --output /zju_0038/
jinjia/workspace/Merging-Scaling-Law-main/eval_outputs/merged_12models/0.2_2 
--batch_size 1 --max_length 2048 --offline
```