#!/bin/bash
# 自动化模型评估脚本
# 自动从GitHub下载代码，从HuggingFace下载模型，并执行评估

set -e

# ========================================
# 用户配置 - 修改这些参数
# ========================================
# HuggingFace模型仓库地址
MODEL_URL="microsoft/DialoGPT-medium"  # 修改为您的模型地址

# 评估数据集 (可选: algebra, analysis, discrete, geometry, number_theory, all)
DATASET="algebra"

# Callback配置
CALLBACK_URL="http://147.8.92.70:22222/api/evaluate/callback"
TASK_ID="eval_task_$(date +%s)"
MODEL_ID="my_model_v1"
BENCHMARK_ID="math_problems"

# 评估参数
BATCH_SIZE=8
MAX_LENGTH=2048

# ========================================
# 自动执行部分 - 无需修改
# ========================================
REPO_URL="https://github.com/maxuan1798/Merging-EVAL.git"
WORK_DIR="./eval_workspace"
REPO_DIR="$WORK_DIR/Merging-EVAL"

# 检测Python命令
PYTHON_CMD="python3"
command -v python3 &> /dev/null || PYTHON_CMD="python"

# 创建工作目录
mkdir -p "$WORK_DIR"

# 克隆或更新代码仓库
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    git pull -q origin main 2>/dev/null || (cd .. && rm -rf Merging-EVAL && git clone -q "$REPO_URL")
    cd - > /dev/null
else
    cd "$WORK_DIR"
    git clone -q "$REPO_URL"
    cd - > /dev/null
fi

# 安装依赖（静默模式）
$PYTHON_CMD -m pip install -q torch transformers pandas tqdm requests swanlab 2>/dev/null || true

# 准备路径
EVAL_SCRIPT="$REPO_DIR/scripts/eval.py"
DATA_DIR="$REPO_DIR/data/eval_partial"
OUTPUT_DIR="$REPO_DIR/output/$TASK_ID"
CACHE_DIR="$REPO_DIR/cache/$TASK_ID"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# 显示配置信息
echo "Starting evaluation..."
echo "Model: $MODEL_URL"
echo "Dataset: $DATASET"
echo "Task ID: $TASK_ID"
echo ""

# 执行评估
if [ "$DATASET" = "all" ]; then
    # 评估所有数据集
    $PYTHON_CMD "$EVAL_SCRIPT" \
        --model "$MODEL_URL" \
        --tokenizer "$MODEL_URL" \
        --dataset "$DATA_DIR" \
        --output "$OUTPUT_DIR" \
        --cache_dir "$CACHE_DIR" \
        --experiment_name "$TASK_ID" \
        --callback_url "$CALLBACK_URL" \
        --task_id "$TASK_ID" \
        --model_id "$MODEL_ID" \
        --benchmark_id "$BENCHMARK_ID" \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --use_swanlab 2>&1 | grep -E "(Evaluation|Callback|Error|✅|❌|Final)"
else
    # 评估单个数据集
    $PYTHON_CMD "$EVAL_SCRIPT" \
        --model "$MODEL_URL" \
        --tokenizer "$MODEL_URL" \
        --file "$DATA_DIR/${DATASET}.json" \
        --output "$OUTPUT_DIR" \
        --cache_dir "$CACHE_DIR" \
        --experiment_name "$TASK_ID" \
        --callback_url "$CALLBACK_URL" \
        --task_id "$TASK_ID" \
        --model_id "$MODEL_ID" \
        --benchmark_id "$BENCHMARK_ID" \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --use_swanlab 2>&1 | grep -E "(Evaluation|Callback|Error|✅|❌|Final)"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed!"
    echo "Results: $OUTPUT_DIR"
else
    echo "Evaluation failed"
    exit 1
fi
