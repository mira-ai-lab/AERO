# synth/inner_loop.py
import argparse
import os
import json
import shutil
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from synth.generator import generate_questions 
from synth.answerer import answer_question, self_correct
from synth.answer_analyzer import (
    cluster_answers_with_model, 
    analyze_distribution, 
    extract_boxed_content,
    check_equivalence_with_model
)
from utils.io import write_jsonl

def process_single_question_self_play(q, model_spec, n_samples=16):
    question_text = q["question"]
    
    # 1. 采样 N 次
    responses = []

    for _ in range(n_samples):
        ans = answer_question(question_text, model_spec, temp=0.7)
        responses.append(ans)
    
    # 2. 模型聚类
    cluster_res = cluster_answers_with_model(responses, model_spec)
    status = analyze_distribution(cluster_res)
    
    ret_data = {
        "kto_data": [],
        "q_gen_label": None,
        "log": {
            "question": question_text, 
            "status": status, 
            "top_clusters": [c["key"] for c in cluster_res["clusters"][:3]]
        }
    }

    if status == "consistent":
        ret_data["q_gen_label"] = "negative" # 太简单
        
    elif status == "chaotic":
        ret_data["q_gen_label"] = "negative" # 太难或格式错误
        
    elif status == "bimodal":
        ret_data["q_gen_label"] = "positive" # 难度适中
        
        # 提取前两个主要簇
        c1 = cluster_res["clusters"][0]
        c2 = cluster_res["clusters"][1]
        
        # 3. 自我攻击 (Self-Correction)
        # 假设 c1 是错的 -> 修正得到 c1_prime
        # 假设 c2 是错的 -> 修正得到 c2_prime
        fix_c1 = self_correct(question_text, c1["example"], model_spec)
        fix_c2 = self_correct(question_text, c2["example"], model_spec)
        
        # 提取修正后的答案
        key_c1_prime = extract_boxed_content(fix_c1)
        key_c2_prime = extract_boxed_content(fix_c2)
        
        # 4. 验证收敛性 (使用模型判断修正后的答案是否一致)
        # 只有当两个路径都修正到了同一个结果，我们才认为找到了真理
        if key_c1_prime and key_c2_prime:
            is_converged = check_equivalence_with_model(key_c1_prime, key_c2_prime, model_spec)
            
            if is_converged:
                ret_data["log"]["convergence"] = "success"
                
                # 认为 key_c1_prime (即 key_c2_prime) 是 Pseudo-Truth
                # 生成 KTO 数据
                # 正样本: 修正后的正确过程 (fix_c1)
                ret_data["kto_data"].append({
                    "prompt": question_text,
                    "completion": fix_c1,
                    "label": True
                })
                
                # 负样本判定：检查 c1 和 c2 谁原本就是错的
                # 再次调用模型判断原始答案 c1 是否等于真理
                c1_is_correct = check_equivalence_with_model(c1["key"], key_c1_prime, model_spec)
                c2_is_correct = check_equivalence_with_model(c2["key"], key_c1_prime, model_spec)
                
                if not c1_is_correct:
                    ret_data["kto_data"].append({
                        "prompt": question_text,
                        "completion": c1["example"],
                        "label": False
                    })
                    
                if not c2_is_correct:
                    ret_data["kto_data"].append({
                        "prompt": question_text,
                        "completion": c2["example"],
                        "label": False
                    })
            else:
                ret_data["log"]["convergence"] = "failed"
                
    return ret_data

def run_inner_loop(n_questions, out_dir, model_spec, round_idx, workers, config_path="config.yaml"):
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        CFG = yaml.safe_load(f)
    n_samples = CFG.get("default", {}).get("sampling_n", 16)
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[InnerLoop] Generating {n_questions} questions...")
    qs = generate_questions(n_questions, model_spec, max_workers=workers)
    
    kto_dataset = []
    q_gen_dataset = [] 
    logs = []
    
    print(f"[InnerLoop] Self-Play processing (N={n_samples} samples/q, Workers={workers})...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交任务
        futures = {
            executor.submit(process_single_question_self_play, q, model_spec, n_samples): q 
            for q in qs
        }
        
        for future in tqdm(as_completed(futures), total=len(qs)):
            try:
                data = future.result()
                if data["kto_data"]:
                    kto_dataset.extend(data["kto_data"])
                
                if data["q_gen_label"]:
                    q_gen_dataset.append({
                        "prompt": futures[future]["prompt"], 
                        "completion": futures[future]["question"],
                        "label": (data["q_gen_label"] == "positive")
                    })
                
                logs.append(data["log"])
            except Exception as e:
                print(f"Error processing question: {e}")

    # 保存
    write_jsonl(os.path.join(out_dir, "kto_data.jsonl"), kto_dataset)
    write_jsonl(os.path.join(out_dir, "q_gen_data.jsonl"), q_gen_dataset)
    write_jsonl(os.path.join(out_dir, "inner_logs.jsonl"), logs)
    
    # 备份到 kto_data 供外部调用
    kto_dir = "kto_data"
    os.makedirs(kto_dir, exist_ok=True)
    shutil.copy(os.path.join(out_dir, "kto_data.jsonl"), os.path.join(kto_dir, "kto_data.jsonl"))
    
    print(f"[InnerLoop] Done. Generated {len(kto_dataset)} KTO samples.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/round_tmp")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--model_spec", type=str, default=None)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10) 
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    ms = args.model_spec or os.environ.get("CURRENT_MODEL")
    run_inner_loop(args.n_questions, args.out_dir, ms, args.round, args.workers, config_path=args.config)