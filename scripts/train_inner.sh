#!/bin/bash
set -euo pipefail

# 显示帮助信息
show_help() {
    cat <<EOF
Usage: $0 [options]
Options:
  --rdzv_endpoint   Rendezvous endpoint (required)
  --nnodes          Number of nodes (required)
  -h, --help        Show this help message
EOF
}

# ============================= 环境配置 =====================================
WORK_DIR="/path/to/project/workspace"
CACHE_DIR="/path/to/cache/directory"
CONDA_ENV_DIR="/path/to/conda/environment"
# ===========================================================================

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rdzv_endpoint)
            RDZV_ENDPOINT="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "${RDZV_ENDPOINT:-}" || -z "${NNODES:-}" ]]; then
    echo "Error: Missing required parameters" >&2
    show_help
    exit 1
fi

# ============================= 环境初始化 ===================================
# 激活conda环境
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_DIR}" || {
    echo "[ERROR] Failed to activate conda environment" >&2
    exit 1
}

# 设置环境变量
export HOME="${WORK_DIR}"
export HF_HOME="${CACHE_DIR}/transformers"
export VLLM_CACHE_DIR="${CACHE_DIR}/vllm"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
export TORCH_HOME="${CACHE_DIR}/torch"
export HF_DATASETS_CACHE="${CACHE_DIR}/dataset"
export TORCH_EXTENSIONS_DIR="${CACHE_DIR}/torch_extension"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_NET_GDR_LEVEL=2

cd "${WORK_DIR}" || {
    echo "[ERROR] Failed to enter work directory" >&2
    exit 1
}
# ===========================================================================

echo "[INFO] Rendezvous Endpoint: ${RDZV_ENDPOINT}"
echo "[INFO] Number of Nodes: ${NNODES}"

# 训练参数配置
base_model=/path/to/pretrained/model
dataset_dir=/path/to/training/dataset
dataset=${dataset_dir}/training_data.json
prompt_key=instruction
response_key=output
SAVE_PATH=/path/to/output/directory

learning_rate=2e-5
num_epochs=5
max_len=32768
batch_size=16
echo "[INFO] Model save path: ${SAVE_PATH}"

# ============================= 启动训练 =====================================
torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    -m openrlhf.cli.train_sft \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --ckpt_path "${SAVE_PATH}" \
    --max_len $max_len \
    --dataset $dataset \
    --input_key $prompt_key \
    --output_key $response_key \
    --apply_chat_template \
    --train_batch_size $batch_size \
    --micro_train_batch_size 1 \
    --max_samples 10000000 \
    --pretrain $base_model \
    --save_path "${SAVE_PATH}" \
    --save_steps 1000 \
    --logging_steps 3 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs $num_epochs \
    --bf16 \
    --flash_attn \
    --learning_rate $learning_rate \
    --gradient_checkpointing \
    --packing_samples \
    --ring_attn_size 2 \
    --ring_head_stride 1 \
    --lr_warmup_ratio 0.05
# ===========================================================================

echo "Training completed successfully"
exit 0