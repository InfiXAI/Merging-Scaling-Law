# 自动化模型评估脚本使用说明

## 快速开始

提供两种方式运行自动化评估：

### 方式1：使用Bash脚本（推荐）

```bash
# 1. 编辑 auto_eval.sh 修改配置
vi auto_eval.sh

# 2. 修改以下参数：
# MODEL_URL="your-huggingface-model"
# DATASET="algebra"  # 或 all
# MODEL_ID="your_model_name"
# BENCHMARK_ID="your_benchmark"

# 3. 运行脚本
./auto_eval.sh
```

### 方式2：使用Python脚本

```bash
# 基本用法
python3 auto_eval.py --model_url microsoft/DialoGPT-medium --dataset algebra

# 完整参数
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset algebra \
  --model_id my_model_v1 \
  --benchmark_id math_problems \
  --batch_size 8 \
  --max_length 2048
```

## 参数说明

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

1. **下载代码**: 自动从GitHub克隆/更新评估代码
2. **下载模型**: 从HuggingFace自动下载指定模型
3. **安装依赖**: 自动安装所需Python包
4. **执行评估**: 调用eval.py进行模型评估
5. **发送回调**: 评估完成后自动发送结果到callback API

## 输出

- 评估结果保存在: `eval_workspace/Merging-EVAL/output/<task_id>/`
- 缓存文件保存在: `eval_workspace/Merging-EVAL/cache/<task_id>/`
- SwanLab可视化: 使用task_id查看实验结果

## 示例

### 评估单个数据集
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset algebra \
  --model_id dialogpt_test
```

### 评估所有数据集
```bash
python3 auto_eval.py \
  --model_url microsoft/DialoGPT-medium \
  --dataset all \
  --model_id dialogpt_full_eval
```

### 使用自定义参数
```bash
python3 auto_eval.py \
  --model_url meta-llama/Llama-2-7b-hf \
  --dataset geometry \
  --model_id llama2_geometry \
  --batch_size 16 \
  --max_length 4096
```

## 注意事项

1. 确保有足够的磁盘空间存储模型（通常5-20GB）
2. 首次运行需要下载模型，时间较长
3. 模型会缓存在本地，后续运行会更快
4. 日志输出已优化，仅显示关键信息
5. 需要联网访问HuggingFace和GitHub

## 故障排除

### 模型下载失败
- 检查网络连接
- 确认HuggingFace模型地址正确
- 某些模型需要登录认证: `huggingface-cli login`

### 依赖安装失败
```bash
pip3 install torch transformers pandas tqdm requests swanlab
```

### 权限错误
```bash
chmod +x auto_eval.sh
chmod +x auto_eval.py
```
