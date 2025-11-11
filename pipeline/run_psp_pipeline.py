#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics Self-Play (PSP) pipeline
多轮自博弈：内循环 (数据合成 + 批评 + 精炼) → 外循环 (DPO 训练)
每轮训练完毕后自动重新部署 vLLM 加载新模型。
"""

import os, json, yaml, subprocess, time, requests
from datetime import datetime
from cluster.cluster_agent import ClusterAgent

# ===== 配置加载 =====
CFG = yaml.safe_load(open("config.yaml"))
STATE_FILE = "pipeline/pipeline_state.json"
VLLM_PORT = 8000

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
    cmd = f"nohup vllm serve {model_path} --port {port} --max-model-len 8192 > vllm_round.log 2>&1 &"
    subprocess.run(cmd, shell=True)
    print(f"[vLLM] 启动命令：{cmd}")

    # 3. 等待启动
    ready = False
    for i in range(40):
        try:
            r = requests.post(f"http://localhost:{port}/generate",
                              json={"prompt": "ping"}, timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(3)
    if ready:
        print(f"[vLLM] ✅ 新模型已上线：http://localhost:{port}/generate\n")
    else:
        print(f"[vLLM] ⚠️ 超时：请检查 vLLM 是否正常启动。\n")

# ===== 内循环 =====
def run_inner_loop(current_model, round_idx):
    print(f"[Round {round_idx}] 🚀 内循环启动（模型：{current_model}）")
    out_dir = f"outputs/round_{round_idx}"
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env["CURRENT_MODEL"] = current_model
    cmd = [
        "python3", "synth/inner_loop.py",
        "--out_dir", out_dir,
        "--n_questions", str(CFG["default"]["questions_per_round"]),
        "--model_spec", current_model
    ]
    subprocess.run(cmd, check=True, env=env)
    subprocess.run(["python3", "dpo/make_dpo_pairs.py"], check=True)
    print(f"[Round {round_idx}] ✅ 内循环完成。\n")

# ===== 外循环 (DPO 训练) =====
def run_outer_loop(current_model, round_idx):
    print(f"[Round {round_idx}] 🧠 外循环 DPO 训练中...")
    out_dir = f"models/psp_round_{round_idx}"
    os.makedirs(out_dir, exist_ok=True)
    cmd_template = CFG["default"]["dpo_train_cmd_template"]
    cmd = cmd_template.format(model=current_model, out_dir=out_dir)
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"[Round {round_idx}] ✅ 外循环训练完成，新模型保存至 {out_dir}")
    return f"local::{out_dir}"

# ===== ClusterAgent =====
def cluster_and_update_prompt(round_idx):
    import json
    path = f"outputs/round_{round_idx}/inner_results.jsonl"
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

# ===== 主流程 =====
def main():
    state = load_state()
    total_rounds = CFG["default"]["rounds"]

    # 首次运行时，部署初始模型
    if state["round"] == 0:
        init_model_path = "/data/gaozhitao/modelhub/Qwen3-1.7B"
        restart_vllm_service(init_model_path, port=VLLM_PORT)
        state["current_model"] = f"http::http://localhost:{VLLM_PORT}/generate"
        save_state(state)

    for r in range(state["round"] + 1, total_rounds + 1):
        cur_model = state["current_model"]
        print(f"\n===== 🌍 Round {r} 启动 (当前模型: {cur_model}) =====")

        # 内循环
        run_inner_loop(cur_model, r)

        # 聚类分析与 prompt 更新
        cluster_and_update_prompt(r)

        # 外循环训练
        new_model_local = run_outer_loop(cur_model, r)

        # 重新部署 vLLM
        new_model_path = new_model_local.replace("local::", "")
        restart_vllm_service(new_model_path, port=VLLM_PORT)

        # 更新状态
        state["round"] = r
        state["current_model"] = f"http::http://localhost:{VLLM_PORT}/generate"
        state["history"].append({
            "round": r,
            "model": new_model_local,
            "timestamp": datetime.now().isoformat()
        })
        save_state(state)

        print(f"✅ Round {r} 完成，vLLM 已更新为新模型。")
        print("============================================\n")

    print("🎯 全部轮次 PSP 训练完成。")

if __name__ == "__main__":
    main()
