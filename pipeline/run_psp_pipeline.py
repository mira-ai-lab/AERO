#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os, json, yaml, subprocess, time, requests, shutil
from datetime import datetime
from utils.io import read_jsonl
# [Change] 导入 KTO 转换工具
from utils.make_kto_data import convert_to_kto_format

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="default_exp")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--gpus", type=str, default=None)
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()

EXP_NAME = args.exp_name
EXP_ROOT = os.path.join("experiments", EXP_NAME)
os.makedirs(EXP_ROOT, exist_ok=True)

print(f"Loading config from: {args.config}")
CFG = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
STATE_FILE = os.path.join(EXP_ROOT, "pipeline_state.json")
VLLM_PORT = args.port
LLAMA_FACTORY_DIR = CFG["default"]["llama_factory_dir"]

if args.gpus:
    TRAIN_GPUS = args.gpus
else:
    TRAIN_GPUS = CFG["default"]["kto_gpus"]

# [Change] 使用 KTO 模板
KTO_TRAIN_TEMPLATE_YAML = os.path.join(LLAMA_FACTORY_DIR, CFG["default"]["kto_train_template_yaml"])
MERGE_TEMPLATE_YAML = os.path.join(LLAMA_FACTORY_DIR, CFG["default"]["kto_merge_template_yaml"])

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    else:
        return {"round": 0, "current_model": CFG["default"]["initial_model"], "history": []}

def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)

def restart_vllm_service(model_path: str, port: int):
    print(f"\n[vLLM] 🔄 Deploying: {model_path}")
    subprocess.run(f"pkill -f 'vllm.*--port {port}' || true", shell=True)
    time.sleep(2)
    
    cmd = (f"CUDA_VISIBLE_DEVICES={TRAIN_GPUS} nohup vllm serve {model_path} "
           f"--port {port} --max-model-len 15360 --tensor-parallel-size 1 "
           f"--gpu-memory-utilization 0.9 --served-model-name psp_model " 
           f"> vllm_{EXP_NAME}.log 2>&1 &")
    subprocess.run(cmd, shell=True)
    
    health_url = f"http://localhost:{port}/health"
    print(f"[vLLM] Waiting for service...")
    for i in range(100):
        try:
            if requests.get(health_url, timeout=3).status_code == 200:
                print(f"[vLLM] ✅ Ready.")
                return
        except:
            pass
        time.sleep(5)
    raise RuntimeError("vLLM failed to start.")

def stop_vllm_service(port: int):
    subprocess.run(f"pkill -f 'vllm.*--port {port}' || true", shell=True)
    time.sleep(5)

def run_inner_loop(current_model, round_idx):
    print(f"[Round {round_idx}] 🚀 Inner Loop (Model: {current_model})")
    out_dir = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}")
    os.makedirs(out_dir, exist_ok=True)
    
    marker = os.path.join(out_dir, "kto_data.jsonl")
    kto_data_dir = os.path.join(EXP_ROOT, "kto_data")
    os.makedirs(kto_data_dir, exist_ok=True)

    if not os.path.exists(marker):
        env = os.environ.copy()
        env["CURRENT_MODEL"] = current_model
        cmd = [
            "python3", "-m", "synth.inner_loop",
            "--out_dir", out_dir,
            "--n_questions", str(CFG["default"]["questions_per_round"]),
            "--model_spec", current_model,
            "--round", str(round_idx),
            "--config", args.config,
            "--workers", "20"
        ]
        subprocess.run(cmd, check=True, env=env)
        
        # Copy results
        shutil.copy(os.path.join(out_dir, "kto_data.jsonl"), os.path.join(kto_data_dir, "kto_data.jsonl"))
    else:
        print("[Skipping generation, data exists]")

    # Convert to KTO format
    convert_to_kto_format(
        os.path.join(kto_data_dir, "kto_data.jsonl"), 
        os.path.join(kto_data_dir, "kto_final.json")
    )

def prepare_kto_data_for_llamafactory(round_idx, llama_factory_dir):
    dataset_name = f"{EXP_NAME}_kto_round_{round_idx}"
    file_name = f"{dataset_name}.json"
    
    # Copy file to LLaMA-Factory data dir
    src = os.path.join(EXP_ROOT, "kto_data", "kto_final.json")
    dst = os.path.join(llama_factory_dir, "data", file_name)
    shutil.copy(src, dst)
    
    # Update dataset_info.json
    info_path = os.path.join(llama_factory_dir, "data", "dataset_info.json")
    try:
        with open(info_path, 'r') as f: info = json.load(f)
    except: info = {}
    
    info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",  
        "columns": {
            "messages": "messages",
            "kto_tag": "label"    
        },
        "tags": {
            "role_tag": "role",        
            "content_tag": "content",   
            "user_tag": "user",         
            "assistant_tag": "assistant"
        }
    }
    
    with open(info_path, 'w') as f: json.dump(info, f, indent=4)
    return dataset_name

def run_outer_loop(base_model_path: str, round_idx: int):
    print(f"[Round {round_idx}] 🧠 KTO Training...")
    
    dataset_name = prepare_kto_data_for_llamafactory(round_idx, LLAMA_FACTORY_DIR)
    lora_output_dir = os.path.join(EXP_ROOT, f"saves/psp_round_{round_idx}")
    final_merged_dir = os.path.join(EXP_ROOT, f"models/psp_round_{round_idx}")
    
    train_yaml_path = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}/kto_config.yaml")
    
    # Load & Modify Train Config
    with open(KTO_TRAIN_TEMPLATE_YAML, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg["model_name_or_path"] = base_model_path
    cfg["dataset"] = dataset_name
    cfg["output_dir"] = lora_output_dir
    # 确保是 KTO stage
    cfg["stage"] = "kto" 
    
    cfg["dataset_dir"] = "LLaMA-Factory/data"
    with open(train_yaml_path, 'w') as f: yaml.dump(cfg, f)
        
    # Train
    cmd_train = f"FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES={TRAIN_GPUS} llamafactory-cli train {train_yaml_path}"
    subprocess.run(cmd_train, shell=True, check=True)
    
    # Merge
    print(f"[Round {round_idx}] 🔄 Merging...")
    merge_yaml_path = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}/merge_config.yaml")
    with open(MERGE_TEMPLATE_YAML, 'r') as f:
        mcfg = yaml.safe_load(f)
    mcfg["model_name_or_path"] = base_model_path
    mcfg["adapter_name_or_path"] = lora_output_dir
    mcfg["export_dir"] = final_merged_dir
    
    with open(merge_yaml_path, 'w') as f: yaml.dump(mcfg, f)
    
    subprocess.run(f"CUDA_VISIBLE_DEVICES={TRAIN_GPUS} llamafactory-cli export {merge_yaml_path}", shell=True, check=True)
    
    # Cleanup LoRA
    if os.path.exists(lora_output_dir): shutil.rmtree(lora_output_dir)
    return f"local::{final_merged_dir}"

def main():
    print(f"🔵 PSP Pipeline (Self-Play KTO) | Exp: {EXP_NAME}")
    state = load_state()
    current_model_path = ""
    
    if state["round"] == 0:
        init_path = CFG["default"]["init_model_path"]
        restart_vllm_service(init_path, VLLM_PORT)
        state["current_model"] = f"http::http://localhost:{VLLM_PORT}"
        current_model_path = init_path
        state["history"].append({"round":0, "model": f"local::{init_path}"})
        save_state(state)
    else:
        last = state["history"][-1]
        current_model_path = last["model"].replace("local::", "")
        restart_vllm_service(current_model_path, VLLM_PORT)

    for r in range(state["round"] + 1, CFG["default"]["rounds"] + 1):
        # 1. Inner Loop
        run_inner_loop(state["current_model"], r)
        
        # 2. Stop vLLM
        stop_vllm_service(VLLM_PORT)
        
        # 3. Outer Loop (KTO)
        new_model_local = run_outer_loop(current_model_path, r)
        
        # 4. Restart vLLM
        new_path = new_model_local.replace("local::", "")
        restart_vllm_service(new_path, VLLM_PORT)
        current_model_path = new_path
        
        # Update State
        state["round"] = r
        state["current_model"] = f"http::http://localhost:{VLLM_PORT}"
        state["history"].append({"round": r, "model": new_model_local})
        save_state(state)
        
    print("🎯 Pipeline Finished.")

if __name__ == "__main__":
    main()