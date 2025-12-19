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

# -------------------- [权重应用和数据复制] --------------------
def apply_weights_and_replicate(kto_data: list, weights: dict) -> list:
    weighted_data = []
    default_weight = 1.0 

    for item in kto_data:
        data_type = item.get("type", "unknown")
        weight = weights.get(data_type, default_weight)
        
        if weight <= 0:
            continue
            
        base_copies = int(weight)
        extra_copy_prob = weight - base_copies 
        
        num_copies = base_copies
        if random.random() < extra_copy_prob:
            num_copies += 1
            
        if num_copies == 0 and weight > 0:
             num_copies = 1 
             
        for _ in range(num_copies):
            weighted_data.append(item)
            
    print(f"[KTO Data] Applied weights. Total samples after replication: {len(weighted_data)}")
    return weighted_data

# -------------------- [数据获取与聚合逻辑] --------------------

def get_data_by_type_from_round(exp_root, round_idx, target_types):
    """
    从指定轮次读取数据，并根据 type 过滤。
    如果 target_types 为 None，则返回该轮次所有数据。
    """
    kto_path = os.path.join(exp_root, f"outputs/round_{round_idx}", "kto_data.jsonl")
    if not os.path.exists(kto_path):
        return []
    
    data = read_jsonl(kto_path)
    if target_types is None:
        return data
        
    filtered = [d for d in data if d.get("type") in target_types]
    return filtered

def aggregate_kto_for_replay(exp_root, history_end_round, replay_pool_size, replay_ratios): 
    """
    聚合历史轮次 (1 到 history_end_round) 的 KTO 数据用于回放。
    """
    all_historical_data = []
    
    # 如果 history_end_round < 1，说明没有历史数据可回放
    if history_end_round < 1:
        print(f"[KTO Replay] No history to replay (History end round: {history_end_round})")
        return []

    # 1. 收集历史数据
    for r in range(1, history_end_round + 1): 
        kto_path = os.path.join(exp_root, f"outputs/round_{r}", "kto_data.jsonl")
        if os.path.exists(kto_path):
            try:
                data = read_jsonl(kto_path)
                all_historical_data.extend(data)
            except Exception as e:
                print(f"Error reading historical KTO data from {kto_path}: {e}")

    if not all_historical_data:
        return []

    # 2. 按类型分组
    historical_groups = {}
    for item in all_historical_data:
        data_type = item.get("type", "unknown")
        if data_type not in historical_groups:
            historical_groups[data_type] = []
        historical_groups[data_type].append(item)
        
    print(f"[KTO Replay] Pool size: {len(all_historical_data)}. Groups: {[f'{k}:{len(v)}' for k, v in historical_groups.items()]}")

    # 3. 根据比例采样
    replay_data = []
    total_ratio = sum(replay_ratios.values())
    
    if total_ratio == 0:
        print("[KTO Replay] Warning: Total replay ratios sum to zero.")
        return []
    
    for data_type, ratio in replay_ratios.items():
        if data_type in historical_groups:
            pool = historical_groups[data_type]
            normalized_ratio = ratio / total_ratio
            target_count = int(replay_pool_size * normalized_ratio)
            actual_count = min(len(pool), target_count)
            
            if actual_count > 0:
                random.shuffle(pool)
                replay_data.extend(pool[:actual_count])
                
    print(f"[KTO Replay] Selected {len(replay_data)} samples from history (1-{history_end_round}).")
    return replay_data

def aggregate_dataset_strategy(exp_root, current_round, config):
    """
    根据配置决定数据组合策略：
    1. Staggered Mode: Gen(N) + Solver(N-1) + Replay(1..N-2)
    2. Normal Mode:    Gen(N) + Solver(N)   + Replay(1..N-1)
    """
    use_staggered = config["default"].get("use_staggered_training", False)
    replay_pool_size = config["default"].get("replay_pool_size", 500)
    replay_ratios = config["default"].get("kto_replay_ratios", {})
    
    fresh_data = []
    history_end_round = 0 
    
    gen_types = ["question_generation", "question_generation_consistent", "question_generation_chaotic"]
    solver_types = ["answer_solver", "answer_refiner"]

    print(f"\n[Data Aggregation] Round {current_round} | Staggered Mode: {use_staggered}")

    # --- A. 获取 Fresh Data (新数据) ---
    if use_staggered:
        # 1. Generator: 总是来自当前轮次 N
        gen_data = get_data_by_type_from_round(exp_root, current_round, gen_types)
        fresh_data.extend(gen_data)
        print(f"  - Generator Data (Round {current_round}): {len(gen_data)} samples")
        
        # 2. Solver: 来自上一轮 N-1
        if current_round > 1:
            solver_data = get_data_by_type_from_round(exp_root, current_round - 1, solver_types)
            fresh_data.extend(solver_data)
            print(f"  - Solver Data (Round {current_round - 1}): {len(solver_data)} samples")
            # Staggered 模式下，Solver 用了 N-1，所以回放只能到 N-2
            history_end_round = current_round - 2
        else:
            print("  - Solver Data: Skipped (Round 1 has no previous solver data)")
            history_end_round = 0
            
    else:
        # Normal Mode: 所有数据都来自当前轮次 N
        current_data = get_data_by_type_from_round(exp_root, current_round, None) # None 表示所有类型
        fresh_data.extend(current_data)
        print(f"  - All Data (Round {current_round}): {len(current_data)} samples")
        # Normal 模式下，当前用了 N，回放可以到 N-1
        history_end_round = current_round - 1

    # --- B. 获取 Replay Data (历史回放) ---
    replay_data = aggregate_kto_for_replay(exp_root, history_end_round, replay_pool_size, replay_ratios)
    
    # --- C. 合并 ---
    final_dataset = fresh_data + replay_data
    print(f"[Data Aggregation] Total: {len(final_dataset)} (Fresh: {len(fresh_data)}, Replay: {len(replay_data)})")
    
    return final_dataset

def run_inner_loop(current_model, round_idx):
    print(f"[Round {round_idx}] 🚀 Inner Loop (Model: {current_model})")
    out_dir = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}")
    os.makedirs(out_dir, exist_ok=True)
    
    marker = os.path.join(out_dir, "inner_logs.jsonl")
    kto_data_dir = os.path.join(EXP_ROOT, "kto_data")
    os.makedirs(kto_data_dir, exist_ok=True)

    # 1. 调用 inner_loop 脚本生成当前轮次数据
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
    else:
        print("[Skipping generation, data exists]")

    # 2. 使用新策略聚合数据 (支持 Staggered + Replay)
    final_dataset = aggregate_dataset_strategy(EXP_ROOT, round_idx, CFG)
    
    # 3. 应用权重并进行数据复制
    kto_weights = CFG["default"].get("kto_weights", {})
    weighted_dataset = apply_weights_and_replicate(final_dataset, kto_weights)

    # 4. 写入文件，供 Outer Loop 读取
    master_kto_path = os.path.join(kto_data_dir, "kto_data.jsonl") 
    write_jsonl(master_kto_path, weighted_dataset)

    # 5. Convert to KTO format
    convert_to_kto_format(
        master_kto_path, 
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
    stop_vllm_service(VLLM_PORT)

if __name__ == "__main__":
    main()