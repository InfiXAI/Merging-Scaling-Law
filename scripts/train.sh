#!/bin/bash

#SBATCH --job-name=deepspeed
#SBATCH --mem=128G
#SBATCH --partition=<PARTITION_NAME>
#SBATCH --account=<ACCOUNT_NAME>
#SBATCH -D .
#SBATCH --output=logs/O-%x.%j
#SBATCH --error=logs/E-%x.%j
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node
#SBATCH --cpus-per-task=160         # 请确认节点实际CPU核心数是否支持该配置
#SBATCH --time=23:59:00             # maximum execution time (HH:MM:SS)

# 获取主节点地址和端口
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=11975
NNODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

# 创建唯一实验名（带时间戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="llamafactory_n${NNODES}_${TIMESTAMP}"

# ============================= 环境配置 =====================================
# 工作目录和conda环境路径
WORK_DIR=<WORK_DIR_PATH>
ORIG_CONDA_ENV=<CONDA_ENV_PATH>
CONDA_ENV_MOUNT=/data/conda_env

# 日志目录配置
LOG_DIR="<LOG_DIR_PATH>/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}" || { echo "Failed to create log directory"; exit 1; }
echo "Log directory: ${LOG_DIR}"
# ===========================================================================

# 容器名称
CONTAINER_IMAGE=<CONTAINER_IMAGE_PATH>
CONTAINER_NAME="llamafactory"
CONTAINER_MOUNT=${ORIG_CONDA_ENV}:${CONDA_ENV_MOUNT},<MOUNT_PATHS>

# 启动训练任务
srun --nodes=${SLURM_NNODES} \
    --container-name=${CONTAINER_NAME} \
    --container-mounts=${CONTAINER_MOUNT} \
    --container-image=${CONTAINER_IMAGE} \
    --container-writable \
    bash -c "
    bash ${WORK_DIR}/train_inner.sh --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} --nnodes ${NNODES}
    "

echo "Job completed successfully"
exit 0
