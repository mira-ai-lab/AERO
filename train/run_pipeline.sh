#!/usr/bin/env bash
set -e

# 示例：运行实验 exp_v1
# 你可以在运行脚本时传入实验名，例如：bash train/run_pipeline.sh exp_v1

EXP_NAME=${1:-"default_exp"}  # 如果不传参数，默认为 default_exp

echo "Starting Experiment: $EXP_NAME"

python3 -u -m pipeline.run_psp_pipeline --exp_name "$EXP_NAME"