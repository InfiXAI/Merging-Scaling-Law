#!/bin/bash
set -euo pipefail

CONDA_ENV=/path/to/project/user/Envs/user_env
eval "$(/path/to/project/miniconda3/bin/conda shell.bash hook)" && conda activate $CONDA_ENV
echo "已激活环境: $CONNA_DEFAULT_ENV"

# —— 1. 实验名称和日志目录 ——
EXPERIMENT_NAME=merge_v1
SLOG_DIR="logs/$EXPERIMENT_NAME"

# —— 2. 基本参数 ——
user=username
WORK_DIR=/path/to/project/user/merging-batch
cd $WORK_DIR
mkdir -p "$SLOG_DIR"
scale="7B"
MODEL_NAME="ModelName-${scale}"
BASE_MODEL="/path/to/models/ModelName/$MODEL_NAME"
SCALING_COEFFICIENT=1.0
MASK_RATE=0
MERGE_METHOD="average_merging"

# —— 定义所有可能的模型路径 ——
MODEL_PATHS=(
  "/path/to/project/user/outputs/model-base-${scale}-v2-subject1"
  "/path/to/project/user/outputs/model-base-${scale}-v2-subject2"
  "/path/to/project/user/outputs/model-base-${scale}-v2-subject3"
  "/path/to/project/user/outputs/model-base-${scale}-v2-subject4"
  "/path/to/project/user/outputs/model-base-${scale}-v2-subject5"
  "/path/to/project/user/outputs/model-base-${scale}-v2-subject6"
  "/path/to/project/user/outputs/model-base-${scale}-v2-subject7"
  "/path/to/project/user/outputs/model-base-${scale}-v2-subject8"
  "/path/to/project/user/outputs/model-base-${scale}-v2-subject9"
)

# —— 检查是否提供了模型数量 ——
if [ $# -eq 0 ]; then
  echo "错误：请提供要合并的模型数量（1-${#MODEL_PATHS[@]}）"
  echo "可用模型路径："
  for i in "${!MODEL_PATHS[@]}"; do
    echo "[$((i+1))] ${MODEL_PATHS[$i]}"
  done
  exit 1
fi

NUM_MODELS_TO_MERGE=$1

# 使用Python生成所有组合，并保存到数组
readarray -t COMBINATION_ARRAY < <(python3 -c "
import itertools
models = list(range(1, ${#MODEL_PATHS[@]} + 1))
combinations = list(itertools.combinations(models, $NUM_MODELS_TO_MERGE))
for comb in combinations:
    print(' '.join(map(str, comb)))
")
echo ${COMBINATION_ARRAY[@]}
TOTAL_COMBINATIONS=${#COMBINATION_ARRAY[@]}
echo "=== 总共需要处理 $TOTAL_COMBINATIONS 种组合 ==="

# 初始化计数器
CURRENT_COUNT=0

# 遍历所有组合
for combination in "${COMBINATION_ARRAY[@]}"; do
  CURRENT_COUNT=$((CURRENT_COUNT + 1))
  echo "=== [进度: $CURRENT_COUNT/$TOTAL_COMBINATIONS] 处理组合: $combination ==="
  
  # —— 根据组合选择模型路径并提取模型名称 ——
  MODELS_TO_MERGE=""
  MODEL_INDICES=""
  for idx in $combination; do
    # 将输入索引（1-based）转换为数组索引（0-based）
    array_idx=$((idx-1))
    if [[ $array_idx -ge 0 && $array_idx -lt ${#MODEL_PATHS[@]} ]]; then
      # 添加模型路径到 MODELS_TO_MERGE
      if [ -z "$MODELS_TO_MERGE" ]; then
        MODELS_TO_MERGE="${MODEL_PATHS[$array_idx]}"
      else
        MODELS_TO_MERGE="$MODELS_TO_MERGE,${MODEL_PATHS[$array_idx]}"
      fi
      # 构建模型索引字符串
      if [ -z "$MODEL_INDICES" ]; then
        MODEL_INDICES="$idx"
      else
        MODEL_INDICES="$MODEL_INDICES-$idx"
      fi
    else
      echo "错误：索引 $idx 无效，可用索引范围为 1 到 ${#MODEL_PATHS[@]}"
      exit 1
    fi
  done
  # —— 验证模型路径是否存在 ——
  IFS=',' read -ra MODEL_ARRAY <<< "$MODELS_TO_MERGE"
  echo ${MODEL_ARRAY[@]}
  for path in "${MODEL_ARRAY[@]}"; do
    if [ ! -d "$path" ]; then
      echo "错误：模型路径不存在: $path"
      exit 1
    fi
  done

  # —— 计算 MODELS_TO_MERGE 中的模型数量 ——
  NUM_MODELS=${#MODEL_ARRAY[@]}

  # 创建输出目录，使用模型数量和组合索引
  OUTPUT_DIR="$WORK_DIR/models/${MODEL_NAME}-cmb/${MERGE_METHOD}_${NUM_MODELS}/sc${SCALING_COEFFICIENT}_r${MASK_RATE}/${MODEL_INDICES}"

  # —— 3. Merge参数 ——
  OPTS=""
  OPTS+=" --merge_method ${MERGE_METHOD}"
  OPTS+=" --output_dir ${OUTPUT_DIR}"
  OPTS+=" --base_model ${BASE_MODEL}"
  OPTS+=" --models_to_merge ${MODELS_TO_MERGE}"
  export TRANSFORMERS_CACHE=/path/to/project/${user}/cache/transformer/
  export TRITON_CACHE_DIR=/path/to/project/${user}/cache/triton/
  export HF_MODULES_CACHE=/path/to/project/${user}/cache/

  # —— 4. 执行任务 ——
  echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 开始merging组合: $combination (${MODEL_INDICES}) ==="
  {
    cd $WORK_DIR &&
    mkdir -p "$OUTPUT_DIR" &&
    /path/to/project/other_user/miniconda3/envs/env_name/bin/python src/main_merging.py \
      $OPTS 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
  } >"$SLOG_DIR/run_${MODEL_INDICES}.log" 2>&1 

  echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 组合 $combination (${MODEL_INDICES}) 任务完成 ==="
  echo "=== 已完成 $CURRENT_COUNT/$TOTAL_COMBINATIONS 个组合 ==="
done

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] 所有组合任务完成 ==="
echo "=== 总共完成了 $TOTAL_COMBINATIONS 种组合的合并 ==="