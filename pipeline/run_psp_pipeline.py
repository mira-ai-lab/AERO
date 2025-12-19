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

# -------------------- [新增函数: 权重应用和数据复制] --------------------
def apply_weights_and_replicate(kto_data: list, weights: dict) -> list:
    weighted_data = []
    
    default_weight = 1.0 

    for item in kto_data:
        data_type = item.get("type", "unknown")
        # 根据配置获取权重
        weight = weights.get(data_type, default_weight)
        
        # 权重小于等于 0 则忽略该数据点
        if weight <= 0:
            continue
            
        # 1. 保证的复制次数 (整数部分)
        base_copies = int(weight)
        
        # 2. 额外复制的概率 (小数部分)
        extra_copy_prob = weight - base_copies 
        
        num_copies = base_copies
        if random.random() < extra_copy_prob:
            num_copies += 1
            
        # 确保原始数据至少被保留一次（如果 weight 介于 0 到 1 之间）
        if num_copies == 0 and weight > 0:
             num_copies = 1 
             
        # 复制数据
        for _ in range(num_copies):
            weighted_data.append(item)
            
    print(f"[KTO Data] Applied weights. Total samples after replication: {len(weighted_data)}")
    return weighted_data

# -------------------- [数据回放聚合 - 保持不变，但增加读取当前轮次数据的健壮性] --------------------
def aggregate_kto_for_replay(exp_root, current_round_idx, replay_pool_size, replay_ratios): 
    """
    聚合所有历史轮次的 KTO 数据，根据 replay_ratios 进行比例采样，并与当前轮次数据合并。
    """
    all_historical_data = []
    
    # 1. 收集所有历史轮次 (1 到 current_round_idx - 1) 的 KTO 数据
    for r in range(1, current_round_idx): 
        kto_path = os.path.join(exp_root, f"outputs/round_{r}", "kto_data.jsonl")
        if os.path.exists(kto_path):
            try:
                data = read_jsonl(kto_path)
                all_historical_data.extend(data)
            except Exception as e:
                print(f"Error reading historical KTO data from {kto_path}: {e}")

    # 2. 按类型分组历史数据
    historical_groups = {}
    for item in all_historical_data:
        data_type = item.get("type", "unknown")
        if data_type not in historical_groups:
            historical_groups[data_type] = []
        historical_groups[data_type].append(item)
        
    print(f"[KTO Data] Historical data collected: {len(all_historical_data)}. Grouped by type: {[f'{k}:{len(v)}' for k, v in historical_groups.items()]}")

    # 3. 根据比例进行采样
    replay_data = []
    total_historical_samples = 0
    sampled_counts = {}

    # 归一化比例
    total_ratio = sum(replay_ratios.values())
    if total_ratio == 0:
        print("[KTO Data] Warning: Total replay ratios sum to zero. No historical data will be replayed.")
    
    for data_type, ratio in replay_ratios.items():
        if data_type in historical_groups and total_ratio > 0:
            pool = historical_groups[data_type]
            normalized_ratio = ratio / total_ratio
            target_count = int(replay_pool_size * normalized_ratio)
            actual_count = min(len(pool), target_count)
            
            if actual_count > 0:
                random.shuffle(pool)
                replay_data.extend(pool[:actual_count])
                total_historical_samples += actual_count
                sampled_counts[data_type] = actual_count
                
    print(f"[KTO Data] Sampled historical data: {total_historical_samples}. Details: {sampled_counts}")
    
    # 4. 收集当前轮次的数据
    current_kto_path = os.path.join(exp_root, f"outputs/round_{current_round_idx}", "kto_data.jsonl")
    # 确保当前轮次的文件不存在时返回空列表
    current_data = read_jsonl(current_kto_path) if os.path.exists(current_kto_path) else [] 
    
    # 5. 合并新数据和回放数据
    final_dataset = current_data + replay_data
    
    print(f"[KTO Data] Total samples before weighting in Round {current_round_idx}: {len(final_dataset)} (New: {len(current_data)}, Replay: {total_historical_samples})")
    
    return final_dataset

# pipeline/run_psp_pipeline.py

def get_data_by_type_from_round(exp_root, round_idx, target_types):
    """
    从指定轮次读取数据，并根据 type 过滤。
    """
    kto_path = os.path.join(exp_root, f"outputs/round_{round_idx}", "kto_data.jsonl")
    if not os.path.exists(kto_path):
        return []
    
    data = read_jsonl(kto_path)
    filtered = [d for d in data if d.get("type") in target_types]
    return filtered

def aggregate_staggered_data(exp_root, current_round_idx, config):
    """
    实现你的机制：
    - Question Data: 来自 Current Round (N)
    - Solver Data: 来自 Previous Round (N-1)
    """
    combined_data = []
    
    # 定义哪些 type 属于 Generator，哪些属于 Solver
    gen_types = ["question_generation", "question_generation_consistent", "question_generation_chaotic"]
    solver_types = ["answer_solver", "answer_refiner"]

    # 1. 获取当前轮 (N) 的 Generator 数据
    # 这部分数据反映了模型在当前能力下对题目难度的探索
    current_gen_data = get_data_by_type_from_round(exp_root, current_round_idx, gen_types)
    combined_data.extend(current_gen_data)
    print(f"[Staggered] Loaded {len(current_gen_data)} Question-Gen samples from Round {current_round_idx}")

    # 2. 获取上一轮 (N-1) 的 Solver 数据
    # 如果是第 1 轮，N-1=0，通常没有输出数据，所以这一步会跳过，符合“第0轮只训练提问能力”
    if current_round_idx > 1:
        prev_round = current_round_idx - 1
        prev_solver_data = get_data_by_type_from_round(exp_root, prev_round, solver_types)
        combined_data.extend(prev_solver_data)
        print(f"[Staggered] Loaded {len(prev_solver_data)} Solver samples from Round {prev_round}")
    else:
        print("[Staggered] Round 1: Skipping solver training (Solver data lag mechanism).")

    # 3. (可选) 依然可以保留 Replay 机制，但要小心不要引入 N 轮的 Solver 数据
    # 如果需要 Replay，建议只 Replay N-2 及之前的 Solver 数据
    
    return combined_data

def run_inner_loop(current_model, round_idx):
    print(f"[Round {round_idx}] 🚀 Inner Loop (Model: {current_model})")
    out_dir = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}")
    os.makedirs(out_dir, exist_ok=True)
    
    marker = os.path.join(out_dir, "inner_logs.jsonl")
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
        
    else:
        print("[Skipping generation, data exists]")

    replay_pool_size = CFG["default"].get("replay_pool_size", 500)
    replay_ratios = CFG["default"].get("kto_replay_ratios", {
        "answer_solver": 0.5, 
        "answer_refiner": 0.3, 
        "question_generation": 0.2
    })

    # # 获取原始数据 (新数据 + 回放数据，按比例采样)
    # raw_kto_dataset = aggregate_kto_for_replay(EXP_ROOT, round_idx, replay_pool_size, replay_ratios)
    
    # # [修复] 3. 应用权重并进行数据复制 (Replication)
    # # 读取权重配置，如果不存在则默认为空字典（即权重为 1.0）
    # kto_weights = CFG["default"].get("kto_weights", {})
    # final_kto_dataset = apply_weights_and_replicate(raw_kto_dataset, kto_weights)

    # # 4. 写入聚合/加权后的数据
    # master_kto_path = os.path.join(kto_data_dir, "kto_data.jsonl")
    # write_jsonl(master_kto_path, final_kto_dataset)

    # # Convert to KTO format
    # convert_to_kto_format(
    #     master_kto_path, 
    #     os.path.join(kto_data_dir, "kto_final.json")
    # )
    staggered_dataset = aggregate_staggered_data(EXP_ROOT, round_idx, CFG)
    
    # 应用权重 (apply_weights_and_replicate 需要确保能处理)
    kto_weights = CFG["default"].get("kto_weights", {})
    final_kto_dataset = apply_weights_and_replicate(staggered_dataset, kto_weights)

    # 写入文件，供 Outer Loop 读取
    master_kto_path = os.path.join(kto_data_dir, "kto_data.jsonl") # 注意这里覆盖了
    write_jsonl(master_kto_path, final_kto_dataset)

    # Convert to KTO format
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
    stop_vllm_service(VLLM_PORT)

if __name__ == "__main__":
    main()