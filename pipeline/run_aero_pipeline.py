#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os, json, yaml, subprocess, time, requests, shutil
from datetime import datetime
import glob
import random
from utils.io import read_jsonl, write_jsonl
from utils.make_kto_data import convert_to_kto_format

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="aero_default_exp")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--gpus", type=str, default=None)
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()

EXP_NAME = args.exp_name
EXP_ROOT = os.path.join("experiments", EXP_NAME)
os.makedirs(EXP_ROOT, exist_ok=True)

print(f"Loading AERO config from: {args.config}")
CFG = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
STATE_FILE = os.path.join(EXP_ROOT, "pipeline_state.json")
VLLM_PORT = args.port
LLAMA_FACTORY_DIR = CFG["default"]["llama_factory_dir"]

if args.gpus:
    TRAIN_GPUS = args.gpus
else:
    TRAIN_GPUS = CFG["default"]["kto_gpus"]

KTO_TRAIN_TEMPLATE_YAML = os.path.join(LLAMA_FACTORY_DIR, CFG["default"]["kto_train_template_yaml"])
MERGE_TEMPLATE_YAML = os.path.join(LLAMA_FACTORY_DIR, CFG["default"]["kto_merge_template_yaml"])

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    else:
        return {"round": 0, "current_model": CFG["default"]["initial_model"], "history": []}

def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)

def restart_vllm_service(model_path: str, base_port: int):
    gpu_list = TRAIN_GPUS.split(",")
    num_instances = len(gpu_list)
    print(f"\n[vLLM] 🔄 Deploying {num_instances} AERO instances...")
    
    for i in range(num_instances):
        port = base_port + i
        subprocess.run(f"pkill -f 'vllm.*--port {port}' || true", shell=True)
    time.sleep(2)
    
    urls = []
    for i, gpu_id in enumerate(gpu_list):
        api_port = base_port + i
        dist_port = 18100 + (api_port - 8000) + i
        internal_port = 19000 + (api_port - 8000) * 10
        
        cmd = (f"CUDA_VISIBLE_DEVICES={gpu_id} "
            f"VLLM_PORT={internal_port} "            
            f"VLLM_DISTRIBUTED_PORT={dist_port} "   
            f"nohup vllm serve {model_path} "
            f"--port {api_port} "                    
            f"--max-model-len 10240 --tensor-parallel-size 1 "
            f"--gpu-memory-utilization 0.9 "
            f"--served-model-name aero_model " 
            f"> vllm_{EXP_NAME}_{api_port}.log 2>&1 &")
        
        subprocess.run(cmd, shell=True)
        urls.append(f"http://localhost:{api_port}")
        print(f"  - Instance {i} started on GPU {gpu_id}, API: {api_port}, DistPort: {dist_port}")
    
    print(f"[vLLM] ⏳ Waiting for all {num_instances} instances to be ready...")
    max_retries = 60 
    for i in range(max_retries):
        ready_count = 0
        for url in urls:
            try:
                resp = requests.get(f"{url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    ready_count += 1
            except:
                continue 
        
        if ready_count == num_instances:
            print(f"[vLLM] ✅ All {num_instances} AERO instances are ready!")
            return ",".join(urls)
        
        print(f"  - Instances ready: {ready_count}/{num_instances}. Still loading... ({i+1}/{max_retries})")
        time.sleep(15)

    raise RuntimeError("One or more vLLM instances failed to start within the timeout period.")

def stop_vllm_service(base_port: int):
    gpu_list = TRAIN_GPUS.split(",")
    for i in range(len(gpu_list)):
        port = base_port + i
        subprocess.run(f"pkill -f 'vllm.*--port {port}' || true", shell=True)
    time.sleep(5)

def apply_weights_and_replicate(kto_data: list, weights: dict) -> list:
    weighted_data = []
    default_weight = 1.0 
    for item in kto_data:
        data_type = item.get("type", "unknown")
        weight = weights.get(data_type, default_weight)
        if weight <= 0: continue
        base_copies = int(weight)
        extra_copy_prob = weight - base_copies 
        num_copies = base_copies + (1 if random.random() < extra_copy_prob else 0)
        if num_copies == 0 and weight > 0: num_copies = 1 
        for _ in range(num_copies):
            weighted_data.append(item)
    print(f"[KTO Data] Applied weights. Total samples: {len(weighted_data)}")
    return weighted_data

def get_data_by_type_from_round(exp_root, round_idx, target_types):
    kto_path = os.path.join(exp_root, f"outputs/round_{round_idx}", "kto_data.jsonl")
    if not os.path.exists(kto_path): return []
    data = read_jsonl(kto_path)
    if target_types is None: return data
    return [d for d in data if d.get("type") in target_types]

def aggregate_kto_for_replay(exp_root, history_end_round, replay_pool_size, replay_ratios): 
    all_historical_data = []
    if history_end_round < 1: return []
    for r in range(1, history_end_round + 1): 
        kto_path = os.path.join(exp_root, f"outputs/round_{r}", "kto_data.jsonl")
        if os.path.exists(kto_path):
            try:
                all_historical_data.extend(read_jsonl(kto_path))
            except: pass
    if not all_historical_data: return []
    historical_groups = {}
    for item in all_historical_data:
        data_type = item.get("type", "unknown")
        historical_groups.setdefault(data_type, []).append(item)
    replay_data = []
    total_ratio = sum(replay_ratios.values())
    if total_ratio == 0: return []
    for data_type, ratio in replay_ratios.items():
        if data_type in historical_groups:
            pool = historical_groups[data_type]
            target_count = int(replay_pool_size * (ratio / total_ratio))
            actual_count = min(len(pool), target_count)
            if actual_count > 0:
                random.shuffle(pool)
                replay_data.extend(pool[:actual_count])
    return replay_data

def aggregate_dataset_strategy(exp_root, current_round, config):
    """
    Staggered Training Strategy
    """
    use_staggered = config["default"].get("use_staggered_training", False)
    replay_pool_size = config["default"].get("replay_pool_size", 500)
    replay_ratios = config["default"].get("kto_replay_ratios", {})
    fresh_data = []
    
    gen_types = ["generator"]
    solver_types = ["solver", "refiner"]
    
    if use_staggered:
        fresh_data.extend(get_data_by_type_from_round(exp_root, current_round, gen_types))
        if current_round > 1:
            fresh_data.extend(get_data_by_type_from_round(exp_root, current_round - 1, solver_types))
            history_end_round = current_round - 2
        else:
            history_end_round = 0
    else:
        fresh_data.extend(get_data_by_type_from_round(exp_root, current_round, None))
        history_end_round = current_round - 1

    replay_data = aggregate_kto_for_replay(exp_root, history_end_round, replay_pool_size, replay_ratios)
    return fresh_data + replay_data

def run_inner_loop(current_model, round_idx):
    print(f"[Round {round_idx}] 🚀 AERO Inner Loop (Model: {current_model})")
    out_dir = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}")
    os.makedirs(out_dir, exist_ok=True)
    kto_data_dir = os.path.join(EXP_ROOT, "kto_data")
    os.makedirs(kto_data_dir, exist_ok=True)

    marker = os.path.join(out_dir, "inner_logs.jsonl")
    if not os.path.exists(marker):
        env = os.environ.copy()
        env["CURRENT_MODEL"] = current_model
        cmd = ["python3", "-m", "synth.inner_loop", "--out_dir", out_dir, "--n_questions", 
               str(CFG["default"]["questions_per_round"]), "--model_spec", current_model, 
               "--round", str(round_idx), "--config", args.config, "--workers", "16"]
        subprocess.run(cmd, check=True, env=env)
    
    final_dataset = aggregate_dataset_strategy(EXP_ROOT, round_idx, CFG)
    weighted_dataset = apply_weights_and_replicate(final_dataset, CFG["default"].get("kto_weights", {}))
    master_kto_path = os.path.join(kto_data_dir, "kto_data.jsonl") 
    write_jsonl(master_kto_path, weighted_dataset)
    convert_to_kto_format(master_kto_path, os.path.join(kto_data_dir, "kto_final.json"))

def prepare_kto_data_for_llamafactory(round_idx, llama_factory_dir):
    dataset_name = f"{EXP_NAME}_kto_round_{round_idx}"
    file_name = f"{dataset_name}.json"
    shutil.copy(os.path.join(EXP_ROOT, "kto_data", "kto_final.json"), os.path.join(llama_factory_dir, "data", file_name))
    info_path = os.path.join(llama_factory_dir, "data", "dataset_info.json")
    try:
        with open(info_path, 'r') as f: info = json.load(f)
    except: info = {}
    info[dataset_name] = {"file_name": file_name, "formatting": "sharegpt", 
                          "columns": {"messages": "messages", "kto_tag": "label"},
                          "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
    with open(info_path, 'w') as f: json.dump(info, f, indent=4)
    return dataset_name

def run_outer_loop(base_model_path: str, round_idx: int):
    dataset_name = prepare_kto_data_for_llamafactory(round_idx, LLAMA_FACTORY_DIR)
    lora_output_dir = os.path.join(EXP_ROOT, f"saves/aero_round_{round_idx}")
    final_merged_dir = os.path.join(EXP_ROOT, f"models/aero_round_{round_idx}")
    train_yaml_path = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}/kto_config.yaml")
    
    with open(KTO_TRAIN_TEMPLATE_YAML, 'r') as f: cfg = yaml.safe_load(f)
    cfg.update({"model_name_or_path": base_model_path, "dataset": dataset_name, "output_dir": lora_output_dir, "stage": "kto", "dataset_dir": "data"})
    with open(train_yaml_path, 'w') as f: yaml.dump(cfg, f)
    subprocess.run(f"FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES={TRAIN_GPUS} llamafactory-cli train {train_yaml_path}", shell=True, check=True)
    
    merge_yaml_path = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}/merge_config.yaml")
    with open(MERGE_TEMPLATE_YAML, 'r') as f: mcfg = yaml.safe_load(f)
    mcfg.update({"model_name_or_path": base_model_path, "adapter_name_or_path": lora_output_dir, "export_dir": final_merged_dir})
    with open(merge_yaml_path, 'w') as f: yaml.dump(mcfg, f)
    subprocess.run(f"CUDA_VISIBLE_DEVICES={TRAIN_GPUS} llamafactory-cli export {merge_yaml_path}", shell=True, check=True)
    if os.path.exists(lora_output_dir): shutil.rmtree(lora_output_dir)
    return f"local::{final_merged_dir}"

def main():
    state = load_state()
    if state["round"] == 0:
        init_path = CFG["default"]["init_model_path"]
        urls = restart_vllm_service(init_path, VLLM_PORT)
        state.update({"current_model": f"http::{urls}", "round": 0})
        state["history"].append({"round":0, "model": f"local::{init_path}"})
        save_state(state)
        current_model_path = init_path
    else:
        current_model_path = state["history"][-1]["model"].replace("local::", "")
        restart_vllm_service(current_model_path, VLLM_PORT)

    for r in range(state["round"] + 1, CFG["default"]["rounds"] + 1):
        run_inner_loop(state["current_model"], r)
        stop_vllm_service(VLL_PORT)
        new_model_local = run_outer_loop(current_model_path, r)
        new_path = new_model_local.replace("local::", "")
        urls = restart_vllm_service(new_path, VLLM_PORT)
        current_model_path = new_path
        state.update({"round": r, "current_model": f"http::{urls}"})
        state["history"].append({"round": r, "model": new_model_local})
        save_state(state)
    stop_vllm_service(VLLM_PORT)

if __name__ == "__main__":
    main()