# synth/inner_loop.py
import argparse
import os
import shutil
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from synth.generator import generate_questions 
from synth.answerer import answer_question, self_correct, ATTACK_SYSTEM_PROMPT
from synth.answer_analyzer import (
    cluster_answers_with_model, 
    analyze_distribution, 
    extract_boxed_content,
    check_equivalence_with_model
)
from utils.io import write_jsonl

def process_single_question_self_play(q, model_spec, n_samples=16):
    """
    对单个问题进行 Self-Play 流程：
    1. 采样 -> 聚类 -> 判定题目质量 (构造出题数据)
    2. 聚类出现分歧 -> 自我修正 -> 验证收敛 (构造解题数据)
    3. 修正过程 -> (构造 Refiner/Critic 数据)
    """
    question_text = q["question"]
    # q["prompt"] 是生成该问题时给模型的指令，用于构造 Question Gen 数据
    q_gen_prompt_text = q.get("prompt", "") 
    
    # --- 1. 采样 N 次 ---
    responses = []
    for _ in range(n_samples):
        # 注意：这里调用 answer_question 内部会加上 ANSWER_SYSTEM_PROMPT
        ans = answer_question(question_text, model_spec, temp=0.9)
        responses.append(ans)
    
    # --- 2. 模型聚类与分布分析 ---
    cluster_res = cluster_answers_with_model(responses, model_spec)
    status = analyze_distribution(cluster_res)
    
    # 容器：用于存放所有生成的 KTO 数据点
    kto_data_points = []
    
    log_info = {
        "question": question_text, 
        "status": status, 
        "top_clusters": [c["key"] for c in cluster_res["clusters"][:3]],
        "convergence": "N/A"
    }

    # ==========================================
    # A. 构造 [出题数据] (Question Generation Data)
    # ==========================================
    # 逻辑：Bimodal (适中) -> Positive; Consistent (太简)/Chaotic (太难) -> Negative
    if q_gen_prompt_text:
        if status == "bimodal": 
            kto_data_points.append({
                "prompt": q_gen_prompt_text,
                "completion": question_text,
                "label": True,
                "type": "question_generation"
            })
        
        if status == "consistent":
            kto_data_points.append({
                "prompt": q_gen_prompt_text,
                "completion": question_text,
                "label": False,
                "type": "question_generation_consistent"
            })
        
        if status == "chaotic":
            kto_data_points.append({
                "prompt": q_gen_prompt_text,
                "completion": question_text,
                "label": False,
                "type": "question_generation_chaotic"
            })

    # ==========================================
    # B. 进入 Self-Correction 流程 (仅针对 Bimodal)
    # ==========================================
    if status == "bimodal":
        # 提取前两个主要簇
        c1 = cluster_res["clusters"][0]
        c2 = cluster_res["clusters"][1]
        
        # 尝试自我修正
        fix_c1 = self_correct(question_text, c1["example"], model_spec)
        fix_c2 = self_correct(question_text, c2["example"], model_spec)
        
        # 提取修正后的答案 Key
        key_c1_prime = extract_boxed_content(fix_c1)
        key_c2_prime = extract_boxed_content(fix_c2)
        
        # 验证收敛性
        if key_c1_prime and key_c2_prime:
            is_converged = check_equivalence_with_model(key_c1_prime, key_c2_prime, model_spec)
            
            if is_converged:
                log_info["convergence"] = "success"
                
                # 此时我们认为 fix_c1 (内容等同 fix_c2) 是“伪真理”(Pseudo-Truth)
                # 也是高质量的推理过程
                
                # --- 检查 c1 和 c2 谁是错的 ---
                c1_is_correct = check_equivalence_with_model(c1["key"], key_c1_prime, model_spec)
                c2_is_correct = check_equivalence_with_model(c2["key"], key_c1_prime, model_spec)
                
                # ==========================================
                # C. 构造 [解题数据] (Answerer Data)
                # ==========================================
                
                # 1. 正样本：问题 -> 修正后的详细推理过程
                kto_data_points.append({
                    "prompt": question_text,
                    "completion": fix_c1,
                    "label": True,
                    "type": "answer_solver"
                })
                
                # 2. 负样本：问题 -> 原始错误答案
                if not c1_is_correct:
                    kto_data_points.append({
                        "prompt": question_text,
                        "completion": c1["example"],
                        "label": False,
                        "type": "answer_solver"
                    })
                
                if not c2_is_correct:
                    kto_data_points.append({
                        "prompt": question_text,
                        "completion": c2["example"],
                        "label": False,
                        "type": "answer_solver"
                    })

                # ==========================================
                # D. 构造 [修正/Refine数据] (Refiner Data)
                # ==========================================
                # 如果 c1 是错的，那么 fix_c1 就是一次成功的“批判与修正”
                # 我们希望模型学会：给定 (问题 + 错误答案) -> 输出 (修正后的过程)
                
                if not c1_is_correct:
                    # 必须手动构建与 synth/answerer.py 中 self_correct 一致的 Prompt
                    refine_prompt = (
                        f"{ATTACK_SYSTEM_PROMPT}\n"
                        f"### Problem\n{question_text}\n\n"
                        f"### Suspected Incorrect Solution\n{c1['example']}\n\n"
                        f"### Your Correction\n"
                    )
                    kto_data_points.append({
                        "prompt": refine_prompt,
                        "completion": fix_c1,
                        "label": True,
                        "type": "answer_refiner"
                    })
                
                if not c2_is_correct:
                    refine_prompt_2 = (
                        f"{ATTACK_SYSTEM_PROMPT}\n"
                        f"### Problem\n{question_text}\n\n"
                        f"### Suspected Incorrect Solution\n{c2['example']}\n\n"
                        f"### Your Correction\n"
                    )
                    # 注意：这里用 fix_c2 (它和 fix_c1 等价，也可以用 fix_c1)
                    kto_data_points.append({
                        "prompt": refine_prompt_2,
                        "completion": fix_c2,
                        "label": True,
                        "type": "answer_refiner"
                    })

            else:
                log_info["convergence"] = "failed"
                
    return {
        "kto_data": kto_data_points,
        "log": log_info
    }

def run_inner_loop(n_questions, out_dir, model_spec, round_idx, workers, config_path="config.yaml"):
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        CFG = yaml.safe_load(f)
    n_samples = CFG.get("default", {}).get("sampling_n", 16)
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[InnerLoop] Generating {n_questions} questions...")
    # generate_questions 返回 list of dict: {"question":..., "prompt":...}
    qs = generate_questions(n_questions, model_spec, max_workers=workers)
    
    all_kto_dataset = []
    logs = []
    
    print(f"[InnerLoop] Self-Play processing (N={n_samples} samples/q, Workers={workers})...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_question_self_play, q, model_spec, n_samples): q 
            for q in qs
        }
        
        for future in tqdm(as_completed(futures), total=len(qs)):
            try:
                data = future.result()
                if data["kto_data"]:
                    all_kto_dataset.extend(data["kto_data"])
                logs.append(data["log"])
            except Exception as e:
                print(f"Error processing question: {e}")

    write_jsonl(os.path.join(out_dir, "kto_data.jsonl"), all_kto_dataset)
    
    write_jsonl(os.path.join(out_dir, "inner_logs.jsonl"), logs)
    
    type_counts = {}
    for item in all_kto_dataset:
        t = item.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        
    print(f"\n[InnerLoop] Done.")
    print(f"Total KTO samples: {len(all_kto_dataset)}")
    print(f"Details: {type_counts}")
    
    # 备份到根目录的 kto_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/round_tmp")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--model_spec", type=str, default=None)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers count") 
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    ms = args.model_spec or os.environ.get("CURRENT_MODEL") or "local::/path/to/model"
    run_inner_loop(args.n_questions, args.out_dir, ms, args.round, args.workers, config_path=args.config)