#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics Self-Play (PSP) pipeline
多轮自博弈：内循环 (数据合成 + 批评 + 精炼) → 外循环 (DPO 训练)
每轮训练完毕后自动重新部署 vLLM 加载新模型。
"""
import argparse
import os, json, yaml, subprocess, time, requests, shutil
from datetime import datetime
from cluster.cluster_agent import ClusterAgent
from utils.io import read_jsonl
from utils.make_dpo_pairs import convert_pairs_to_sharegpt

# ===== 1. 获取实验名称 =====
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="default_exp", help="实验名称，用于隔离数据和模型")
parser.add_argument("--port", type=int, default=8001, help="vLLM 服务端口")
args = parser.parse_args()

EXP_NAME = args.exp_name
# 所有该实验的数据都放在 experiments/{EXP_NAME}/ 下
EXP_ROOT = os.path.join("experiments", EXP_NAME)
os.makedirs(EXP_ROOT, exist_ok=True)

# ===== 配置加载 =====
CFG = yaml.safe_load(open("config.yaml"))
STATE_FILE = "pipeline/pipeline_state.json"
VLLM_PORT = args.port
# LLaMA-Factory 相关配置
LLAMA_FACTORY_DIR = CFG["default"]["llama_factory_dir"]
DPO_GPUS = CFG["default"]["dpo_gpus"]
DPO_TRAIN_TEMPLATE_YAML = os.path.join(LLAMA_FACTORY_DIR, CFG["default"]["dpo_train_template_yaml"])
DPO_MERGE_TEMPLATE_YAML = os.path.join(LLAMA_FACTORY_DIR, CFG["default"]["dpo_merge_template_yaml"])


# ===== 状态管理 =====
def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    else:
        return {
            "round": 0,
            "current_model": CFG["default"]["initial_model"],
            "history": []
        }

def save_state(state):
    json.dump(state, open(STATE_FILE, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

# ===== vLLM 部署 =====
def restart_vllm_service(model_path: str, port: int = 8000):
    """
    重启 vLLM 服务，使其加载新的模型。
    默认假设 vLLM 命令可用：vllm serve <model_path> --port <port>
    """
    print(f"\n[vLLM] 🔄 准备重新部署模型: {model_path}")
    # 1. 停止旧进程
    subprocess.run(f"pkill -f 'vllm.*--port {port}' || true", shell=True)

    # 2. 启动新模型
    vllm_gpus = "0,1" 
    tensor_parallel_size = 2
    cmd = (f"CUDA_VISIBLE_DEVICES={vllm_gpus} nohup vllm serve {model_path} "
           f"--port {port} --max-model-len 8192 --tensor-parallel-size {tensor_parallel_size} --gpu-memory-utilization 0.95 "
           f"--served-model-name psp_model " 
           f"> vllm_round.log 2>&1 &")
    subprocess.run(cmd, shell=True)
    print(f"[vLLM] 启动命令：{cmd}")

    # 3. 等待启动
    ready = False
    health_url = f"http://localhost:{port}/health"
    print(f"[vLLM] 正在等待服务启动 (GET {health_url})...")

    for i in range(100):
        try:
            # 使用 GET 请求访问 vLLM 的 /health 端点
            r = requests.get(health_url, timeout=3)
            if r.status_code == 200:
                ready = True
                break
            else:
                print(f"[vLLM] ... (状态: {r.status_code})")
                time.sleep(5)
        except requests.exceptions.ConnectionError:
            print("[vLLM] ... (连接被拒绝，vLLM 尚未启动)")
            time.sleep(3)
        except Exception as e:
            print(f"[vLLM] ... (发生错误: {e})")
            time.sleep(5)

    if ready:
        print(f"[vLLM] ✅ 新模型已上线：http://localhost:{port}\n")
    else:
        print(f"[vLLM] ⚠️ 超时：请检查 vLLM 是否正常启动 (查看 vllm_round.log)。\n")
        # [重要] 抛出异常以停止流水线
        raise RuntimeError("vLLM service failed to start.")

def stop_vllm_service(port: int = 8000):
    """
    显式停止 vLLM 服务以释放 GPU 内存。
    """
    print(f"\n[vLLM] 🛑 停止 vLLM 服务 (Port: {port}) 以释放 GPU 资源...")
    # 1. 停止进程
    subprocess.run(f"pkill -f 'vllm.*--port {port}' || true", shell=True)
    # 给予一些时间确保进程完全退出
    time.sleep(5) 
    print(f"[vLLM] ✅ 服务已停止。")

# ===== 内循环 =====
def run_inner_loop(current_model, round_idx):
    print(f"[Round {round_idx}] 🚀 内循环启动（模型：{current_model}）")
    out_dir = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}")
    os.makedirs(out_dir, exist_ok=True)

    # [新增] 检查点逻辑：如果结果文件已存在，则跳过生成
    marker_file = os.path.join(out_dir, "inner_results.jsonl")
    run_generation = True

    dpo_data_dir = os.path.join(EXP_ROOT, "dpo_data")
    os.makedirs(dpo_data_dir, exist_ok=True)
    
    if os.path.exists(marker_file):
        print(f"[Round {round_idx}] ⚠️ 检测到内循环数据已存在: {marker_file}")
        print(f"[Round {round_idx}] ⏭️ 跳过数据生成阶段，直接恢复数据状态...")
        run_generation = False
        
        # 恢复数据逻辑
        files_to_restore = ["answers_pairs.jsonl", "questions_pairs.jsonl", "critic_pairs.jsonl"]
        for fname in files_to_restore:
            src = os.path.join(out_dir, fname)
            dst = os.path.join(dpo_data_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)

    if run_generation:
        env = os.environ.copy()
        env["CURRENT_MODEL"] = current_model
        cmd = [
            "python3", "-m", "synth.inner_loop",
            "--out_dir", out_dir,
            "--n_questions", str(CFG["default"]["questions_per_round"]),
            "--model_spec", current_model,
            "--round", str(round_idx) # [新增] 传递轮次信息
        ]
        subprocess.run(cmd, check=True, env=env)

        files_to_copy = ["answers_pairs.jsonl", "questions_pairs.jsonl", "critic_pairs.jsonl"]
        for fname in files_to_copy:
            src = os.path.join(out_dir, fname)
            dst = os.path.join(dpo_data_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)
    
    print(f"[Round {round_idx}] Converting to ShareGPT format...")
    
    pairs_map = {
        "answers_pairs.jsonl": "answers_dpo.json",
        "questions_pairs.jsonl": "questions_dpo.json",
        "critic_pairs.jsonl": "critic_dpo.json"
    }
    
    for input_name, output_name in pairs_map.items():
        inp_path = os.path.join(dpo_data_dir, input_name)
        out_path = os.path.join(dpo_data_dir, output_name)
        if os.path.exists(inp_path):
            convert_pairs_to_sharegpt(inp_path, out_path)

    print(f"[Round {round_idx}] ✅ 内循环准备完成。\n")


# ===== DPO 数据集准备 (LLaMA-Factory) =====
def prepare_dpo_data_for_llamafactory(round_idx, llama_factory_dir):
    dataset_name = f"{EXP_NAME}_dpo_round_{round_idx}"
    file_name = f"{dataset_name}.json"
    
    dataset_file_path = os.path.join(llama_factory_dir, "data", file_name)
    dataset_info_path = os.path.join(llama_factory_dir, "data", "dataset_info.json")
    
    dpo_data_dir = os.path.join(EXP_ROOT, "dpo_data")
    
    combined_data = []
    for fname in ["answers_dpo.json", "questions_dpo.json", "critic_dpo.json"]:
        fpath = os.path.join(dpo_data_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                combined_data.extend(json.load(f))
            
    with open(dataset_file_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    try:
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    except Exception:
        dataset_info = {}

    dataset_info[dataset_name] = {
        "file_name": file_name,
        "ranking": True,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"}
    }
    
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)
        
    return dataset_name


# ===== 外循环 (DPO 训练 - LoRA 方式) =====
def run_outer_loop(base_model_path: str, round_idx: int):
    """
    执行 LLaMA-Factory LoRA DPO 训练与合并。
    训练完成后，删除 LoRA 适配器和检查点，只保留最终合并的模型。
    """
    dataset_name = prepare_dpo_data_for_llamafactory(round_idx, LLAMA_FACTORY_DIR)
    
    # [修改] LoRA 和 Merge 路径基于 EXP_ROOT
    lora_output_dir = os.path.join(EXP_ROOT, f"saves/psp_round_{round_idx}")
    final_merged_model_dir = os.path.join(EXP_ROOT, f"models/psp_round_{round_idx}")
    
    # Config 文件也保存到实验目录
    dynamic_train_yaml_path = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}/dpo_train_config.yaml")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(dynamic_train_yaml_path), exist_ok=True)

    with open(DPO_TRAIN_TEMPLATE_YAML, 'r', encoding='utf-8') as f:
        train_config = yaml.safe_load(f)
        
    train_config["model_name_or_path"] = base_model_path
    train_config["dataset"] = dataset_name
    train_config["output_dir"] = lora_output_dir
    train_config["dataset_dir"] = os.path.join(LLAMA_FACTORY_DIR, "data")
    
    with open(dynamic_train_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(train_config, f)
        
    # 3. 执行 DPO 训练命令
    cmd_train = (f"FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES={DPO_GPUS} "
                 f"llamafactory-cli train {dynamic_train_yaml_path}")
    print(f"[RUN] {cmd_train}")
    subprocess.run(cmd_train, shell=True, check=True)
    print(f"[Round {round_idx}] ✅ DPO 训练 (LoRA) 完成. 适配器保存在 {lora_output_dir}")

    # 4. 动态配置模型合并 YAML
    print(f"[Round {round_idx}] 🔄 合并模型中...")
    final_merged_model_dir = f"models/psp_round_{round_idx}" # 最终完整模型路径
    dynamic_merge_yaml_path = f"outputs/round_{round_idx}/merge_config.yaml"
    
    with open(DPO_MERGE_TEMPLATE_YAML, 'r', encoding='utf-8') as f:
        merge_config = yaml.safe_load(f)
        
    merge_config["model_name_or_path"] = base_model_path
    merge_config["adapter_name_or_path"] = lora_output_dir
    merge_config["export_dir"] = final_merged_model_dir

    with open(dynamic_merge_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(merge_config, f)

    # 5. 执行模型合并命令
    cmd_merge = f"CUDA_VISIBLE_DEVICES={DPO_GPUS} llamafactory-cli export {dynamic_merge_yaml_path}"
    print(f"[RUN] {cmd_merge}")
    subprocess.run(cmd_merge, shell=True, check=True)
    
    print(f"[Round {round_idx}] ✅ 模型合并完成，新模型保存至 {final_merged_model_dir}")

    # =====================================================
    # 11/18 清理 LoRA 权重和检查点
    # =====================================================
    if os.path.exists(lora_output_dir):
        print(f"[Cleanup] 🗑️ 正在删除 LoRA 中间产物 (节省空间): {lora_output_dir}")
        try:
            shutil.rmtree(lora_output_dir)
            print(f"[Cleanup] ✅ 已删除 {lora_output_dir}")
        except Exception as e:
            print(f"[Cleanup] ⚠️ 删除失败: {e}")
    # =====================================================
    return f"local::{final_merged_model_dir}"


# ===== ClusterAgent =====
def cluster_and_update_prompt(round_idx):
    import json
    path = os.path.join(EXP_ROOT, f"outputs/round_{round_idx}/inner_results.jsonl")
    if not os.path.exists(path):
        print("⚠️ 未找到 inner_results.jsonl，跳过 cluster 分析。")
        return
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                questions.append(j.get("question",""))
    ca = ClusterAgent(n_clusters=CFG["default"]["cluster"]["n_clusters"])
    res = ca.analyze_and_suggest(
        questions,
        entropy_threshold=CFG["default"]["cluster"]["entropy_threshold"]
    )
    if res.get("suggestion"):
        print("[ClusterAgent] 🔄 更新生成器 prompt：", res["suggestion"]["prompt_suggestion"])
        ca.apply_suggestion_to_prompt(res["suggestion"], "synth/prompt_template.txt")
    else:
        print("[ClusterAgent] ✅ 问题分布良好，无需修改 prompt。")
    with open(f"outputs/round_{round_idx}/cluster_report.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

# ===== 主流程 (修改) =====
def main():
    print(f"🔵 启动 PSP Pipeline | 实验名称: {EXP_NAME}")
    print(f"📂 实验根目录: {EXP_ROOT}")

    state = load_state()
    total_rounds = CFG["default"]["rounds"]
    
    current_model_path = "" # (新) 跟踪当前模型的 *文件路径*

    # 首次运行时，部署初始模型
    if state["round"] == 0:
        init_model_path = CFG["default"]["init_model_path"]
        restart_vllm_service(init_model_path, port=VLLM_PORT)
        state["current_model"] = f"http::http://localhost:{VLLM_PORT}"
        
        state["history"].append({
            "round": 0,
            "model": f"local::{init_model_path}",
            "timestamp": datetime.now().isoformat()
        })
        current_model_path = init_model_path 
        save_state(state)
    else:
        # === [修复] 断点续训逻辑 ===
        # 如果不是首次运行，从 history 加载最新的模型路径，并重新部署 vLLM
        if not state["history"]:
            raise ValueError("State shows round > 0 but history is empty!")
            
        last_model_record = state["history"][-1]
        current_model_path = last_model_record["model"].replace("local::", "")
        
        print(f"⚠️ [Resume] 检测到中断状态 (Round {state['round']})。")
        print(f"🔄 正在恢复部署上一轮的模型: {current_model_path}")
        
        # 这一步是关键：必须在进入循环前把服务拉起来
        restart_vllm_service(current_model_path, port=VLLM_PORT)
        
        # 确保内存中的 state URL 是正确的
        state["current_model"] = f"http::http://localhost:{VLLM_PORT}"
        # ===========================

    for r in range(state["round"] + 1, total_rounds + 1):
        cur_model_endpoint = state["current_model"] 
        print(f"\n===== 🌍 Round {r} 启动 (当前模型: {cur_model_endpoint}) =====")
        print(f"本轮 DPO 训练将基于模型路径: {current_model_path}")

        # 内循环 (使用 vLLM endpoint)
        run_inner_loop(cur_model_endpoint, r)

        # 聚类分析与 prompt 更新
        # cluster_and_update_prompt(r)

        # 停止 vLLM 以释放 GPU
        print(f"[Round {r}] 释放 GPU：准备停止 vLLM 服务...")
        stop_vllm_service(port=VLLM_PORT)
        print(f"[Round {r}] GPU 已释放，准备 DPO 训练...")

        # 外循环训练
        new_model_local = run_outer_loop(current_model_path, r)

        # 重新部署 vLLM
        new_model_path = new_model_local.replace("local::", "")
        restart_vllm_service(new_model_path, port=VLLM_PORT)

        # 更新路径
        current_model_path = new_model_path 
        
        # 更新状态
        state["round"] = r
        state["current_model"] = f"http::http://localhost:{VLLM_PORT}"
        state["history"].append({
            "round": r,
            "model": new_model_local,
            "timestamp": datetime.now().isoformat()
        })
        save_state(state)

        print(f"✅ Round {r} 完成，vLLM 已更新为新模型。")
        print("============================================\n")

    print("🎯 全部轮次 PSP 训练完成。")
    # 训练结束后也可以选择停止服务
    stop_vllm_service(port=VLLM_PORT)
if __name__ == "__main__":
    main()