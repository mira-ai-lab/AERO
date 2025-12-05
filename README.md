## How to run
```bash
nohup bash train/run_pipeline.sh {experiment_name} {VLLM_PORT} {GPUs} --config {config.yaml} > psp_{experiment_name}.log 2>&1 &
```

## How to eval using UGPhysics
require antlr4-python3-runtime==4.11
```bash
CUDA_VISIBLE_DEVICES=3 python codes/generate_open.py \
    --model "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_R1_10_1000_1202/models/psp_round_1" \
    --system 'Please reason step by step, and put your final answer within \boxed{}.' \
    --subject "all" \
    --tensor_parallel_size 1
```

```bash
python codes/eval.py --model_path "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_R1_10_1000_1202/models/psp_round_1" --subject "all"
```