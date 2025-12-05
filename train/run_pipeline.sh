#!/usr/bin/env bash
set -e

EXP_NAME=${1:-"default_exp"}
PORT=${2:-8001}
GPUS=${3:-"4,5"}  # [新增] 第3个参数为 GPU ID，默认 4,5

echo "🚀 Starting Experiment: $EXP_NAME"
echo "🔌 Port: $PORT | 🎮 GPUs: $GPUS"

# 传递 --gpus 参数
python3 -u -m pipeline.run_psp_pipeline \
    --exp_name "$EXP_NAME" \
    --port "$PORT" \
    --gpus "$GPUS" \
    --config "$CONFIG_PATH"