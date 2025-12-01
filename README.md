## How to run
```bash
nohup bash train/run_pipeline.sh {experiment_name} {VLLM_PORT} > psp_{experiment_name}.log 2>&1 &
```

## How to specify the GPU

modify config.yaml the key of  "dpo_gpus" (which is also vllm infer gpus)