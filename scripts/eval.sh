#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=11
#SBATCH --nodelist=<CLUSTER_NODE_RANGE>
#SBATCH --mem=256G
#SBATCH --partition=<PARTITION_NAME>
#SBATCH --account=<ACCOUNT_NAME>
#SBATCH --output=logs/O-%x.%j

model=$1

export PYTHONPATH=<PROJECT_PATH>/mergekit:$PYTHONPATH

cd <PROJECT_WORKDIR>

CUDA_VISIBLE_DEVICES=0 <PYTHON_ENV_PATH>/bin/python eval.py --model $model