#!/usr/bin/env bash
set -e

EXP_NAME=${1:-"default_exp"}
PORT=${2:-8001}
GPUS=${3:-"4,5"}
CONFIG_PATH=${4:-"config.yaml"}

echo "🚀 Starting Experiment: $EXP_NAME"
echo "🔌 Port: $PORT | 🎮 GPUs: $GPUS"

python3 -u -m pipeline.run_aero_pipeline \
    --exp_name "$EXP_NAME" \
    --port "$PORT" \
    --gpus "$GPUS" \
    --config "$CONFIG_PATH"