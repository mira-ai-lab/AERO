## How to run
```bash
nohup bash train/run_pipeline.sh > psp_main.log 2>&1 &
```

## How to specify the GPU

modify config.yaml the key of  "dpo_gpus"

and PSP/pipeline/run_psp_pipeline.py "vllm_gpus"