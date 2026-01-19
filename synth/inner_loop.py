import argparse
import os
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
    question_text = q["question"]
    q_gen_prompt_text = q.get("prompt", "") 
    
    responses = []
    with ThreadPoolExecutor(max_workers=n_samples) as sample_executor:
        futures = [
            sample_executor.submit(answer_question, question_text, model_spec, temp=1.0)
            for _ in range(n_samples)
        ]
        responses = [f.result() for f in futures]
    
    cluster_res = cluster_answers_with_model(responses, model_spec)
    status = analyze_distribution(cluster_res)
    
    kto_data_points = []
    
    log_info = {
        "question": question_text, 
        "status": status, 
        "top_clusters": [c["key"] for c in cluster_res["clusters"][:3]],
        "convergence": "N/A"
    }

    if q_gen_prompt_text:
        kto_data_points.append({
            "prompt": q_gen_prompt_text,
            "completion": question_text,
            "label": status == "zpd",
            "type": "generator"
        })


    if status == "zpd" and len(cluster_res["clusters"]) >= 2:
        c1 = cluster_res["clusters"][0]
        c2 = cluster_res["clusters"][1]
        
        fix_c1 = self_correct(question_text, c1["example"], model_spec)
        fix_c2 = self_correct(question_text, c2["example"], model_spec)
        
        key_c1_prime = extract_boxed_content(fix_c1)
        key_c2_prime = extract_boxed_content(fix_c2)
        
        if key_c1_prime and key_c2_prime:
            is_converged = check_equivalence_with_model(key_c1_prime, key_c2_prime, model_spec)
            
            if is_converged:
                log_info["convergence"] = "success"
                c1_is_correct = check_equivalence_with_model(c1["key"], key_c1_prime, model_spec)
                c2_is_correct = check_equivalence_with_model(c2["key"], key_c1_prime, model_spec)
                
                kto_data_points.append({
                    "prompt": question_text,
                    "completion": fix_c1,
                    "label": True,
                    "type": "solver"
                })
                
                if not c1_is_correct:
                    kto_data_points.append({
                        "prompt": question_text,
                        "completion": c1["example"],
                        "label": False,
                        "type": "solver"
                    })
                
                if not c2_is_correct:
                    kto_data_points.append({
                        "prompt": question_text,
                        "completion": c2["example"],
                        "label": False,
                        "type": "solver"
                    })

                if not c1_is_correct:
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
                        "type": "refiner"
                    })
                
                if not c2_is_correct:
                    refine_prompt_2 = (
                        f"{ATTACK_SYSTEM_PROMPT}\n"
                        f"### Problem\n{question_text}\n\n"
                        f"### Suspected Incorrect Solution\n{c2['example']}\n\n"
                        f"### Your Correction\n"
                    )
                    kto_data_points.append({
                        "prompt": refine_prompt_2,
                        "completion": fix_c2,
                        "label": True,
                        "type": "refiner"
                    })

            else:
                log_info["convergence"] = "failed"
                
    return {
        "kto_data": kto_data_points,
        "log": log_info
    }

def run_inner_loop(n_questions, out_dir, model_spec, round_idx, workers, config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        CFG = yaml.safe_load(f)
    n_samples = CFG.get("default", {}).get("sampling_n", 16)
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[InnerLoop] Generating {n_questions} questions...")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/round_tmp")
    parser.add_argument("--n_questions", type=int, default=1000)
    parser.add_argument("--model_spec", type=str, default=None)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers count") 
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    ms = args.model_spec or os.environ.get("CURRENT_MODEL") or "local::/path/to/model"
    run_inner_loop(args.n_questions, args.out_dir, ms, args.round, args.workers, config_path=args.config)