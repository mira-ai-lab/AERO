# synth/inner_loop.py
import argparse
import os
import json
import shutil
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from synth.generator import generate_questions 
from synth.answerer import answer_question, self_refine
from synth.critic import verify_answer, check_answer_equivalence, generate_critique_with_gold
from utils.io import write_jsonl
from tqdm import tqdm
from collections import defaultdict

# CFG_PATH = "config.yaml"
# if os.path.exists(CFG_PATH):
#     with open(CFG_PATH, 'r', encoding='utf-8') as f:
#         CFG = yaml.safe_load(f)
# else:
#     CFG = {}

def run_refinement_loop(question, initial_answer, critic_model_spec, ground_truth, use_gold_for_critique=False, max_refine=3):
    """
    运行一次【批评 -> 精炼】轨迹。
    """
    current_answer = initial_answer
    trajectory = []
    status = "failed_initial"
    final_answer = current_answer
    
    # 获取初始 Critique (仅用于生成 Feedback，不用于真值判断)
    if use_gold_for_critique:
        # Teacher Mode: 看着答案批评
        critique_data = generate_critique_with_gold(question, current_answer, ground_truth, critic_model_spec)
    else:
        # Student Mode: 盲批
        critique_data = verify_answer(current_answer, question, model_spec=critic_model_spec)
    
    initial_critique_raw = critique_data["raw_output"]
    initial_critique_prompt = critique_data["prompt"]

    for i in range(max_refine):
        # 1. 提取 Feedback
        issues = critique_data.get("issues", []) or critique_data.get("critic_details", {}).get("critical_errors", [])
        feedback = "\n".join(issues[:3]) or "Please check calculations and logic."
        
        # 2. Self-Refine
        refiner_spec = os.environ.get("CURRENT_MODEL") or critic_model_spec 
        current_answer = self_refine(question, current_answer, feedback, model_spec=refiner_spec)
        
        # 3. 验证 (Local Check against Oracle Gold)
        check_res = check_answer_equivalence(question, ground_truth, current_answer, model_spec=refiner_spec)
        is_passed = check_res["passed"]
        
        trajectory.append({
            "step": i+1, 
            "feedback": feedback, 
            "answer": current_answer, 
            "passed": is_passed
        })
        
        if is_passed:
            status = f"refined_pass_{i+1}"
            final_answer = current_answer
            break
            
        # 4. 如果还没对，生成下一轮 Critique 用于继续修正
        if i < max_refine - 1:
            if use_gold_for_critique:
                critique_data = generate_critique_with_gold(question, current_answer, ground_truth, critic_model_spec)
            else:
                critique_data = verify_answer(current_answer, question, model_spec=critic_model_spec)
            
    return {
        "final_answer": final_answer,
        "status": status,
        "initial_critique_raw": initial_critique_raw,
        "initial_critique_prompt": initial_critique_prompt,
        "trajectory": trajectory
    }

def process_single_question(q, model_spec, oracle_model, max_refine, round_idx, warmup_rounds):
    question_text = q["question"]
    question_prompt = q.get("prompt", "")
    ret = {"result": None, "answer_pair": None, "question_candidate": None, "critic_pair": None}

    try:
        # 1. 初始作答 (S0)
        initial_answer = answer_question(question_text, model_spec=model_spec)
        
        # 2. Oracle 判决 (S0 Check) - 获取真值和标准答案
        s0_check = verify_answer(initial_answer, question_text, model_spec=oracle_model)
        s0_is_correct = s0_check["passed"]
        oracle_gold = s0_check.get("correct_solution")

        # 补救：如果 Oracle 判错但没给 GT (罕见)，尝试强制生成一次
        if not s0_is_correct and not oracle_gold:
            #  print(f"  [Skip] Oracle failed to provide GT for Q: {question_text[:10]}...")
             return ret

        final_record = None

        if s0_is_correct:
            # Case A: Easy Sample
            final_record = {
                "question": question_text, "initial_answer": initial_answer, 
                "final_answer": initial_answer, "status": "passed_initial", 
                "critic_report": s0_check["critic_details"]
            }
            ret["question_candidate"] = {"question": question_text, "label": "easy", "prompt": question_prompt}

        else:
            # Case B: Hard Sample
            ret["question_candidate"] = {"question": question_text, "label": "hard", "prompt": question_prompt}
            
            if not oracle_gold:
                return ret # 无法获取 GT，放弃此题

            is_warmup = round_idx <= warmup_rounds
            
            if is_warmup:
                # === Warmup: Teacher Mode (Gold Assist) ===
                run = run_refinement_loop(
                    question_text, initial_answer, 
                    critic_model_spec=model_spec, 
                    ground_truth=oracle_gold, 
                    use_gold_for_critique=True, 
                    max_refine=max_refine
                )
                
                if "pass" in run["status"]:
                    # Answer DPO: Refined > Initial
                    ret["answer_pair"] = {"question": question_text, "chosen": run["final_answer"], "rejected": initial_answer}
                    final_record = {"question": question_text, "final_answer": run["final_answer"], "status": run["status"]}
                else:
                    # 即使有老师教也没学会，放弃
                    final_record = {"question": question_text, "final_answer": run["final_answer"], "status": "failed_with_teacher"}

            else:
                # === Post-Warmup: Sampling Mode (Self-Correction) ===
                N_SAMPLES = 16
                successful_runs = []
                failed_runs = []
                
                # 采样 N 次尝试
                for _ in range(N_SAMPLES):
                    run = run_refinement_loop(
                        question_text, initial_answer, 
                        critic_model_spec=model_spec, 
                        ground_truth=oracle_gold, 
                        use_gold_for_critique=False, # 盲批
                        max_refine=1    #这里是为了保证critic的prompt一致 只refine一次。
                    )
                    
                    if "pass" in run["status"]:
                        successful_runs.append(run)
                    else:
                        failed_runs.append(run)
                    
                    # 只要凑齐一对正负样本即可提前结束
                    if successful_runs and failed_runs:
                        break
                
                if successful_runs:
                    best = successful_runs[0]
                    ret["answer_pair"] = {"question": question_text, "chosen": best["final_answer"], "rejected": initial_answer}
                    final_record = {"question": question_text, "final_answer": best["final_answer"], "status": best["status"]}
                    
                    if failed_runs:
                        worst = failed_runs[0]
                        # Critic DPO: 能修好的 Critique vs 修不好的 Critique
                        ret["critic_pair"] = {
                            "prompt": best["initial_critique_prompt"],
                            "chosen": best["initial_critique_raw"],
                            "rejected": worst["initial_critique_raw"]
                        }
                else:
                    final_record = {"question": question_text, "final_answer": failed_runs[0]["final_answer"], "status": "failed_all_samples"}

        if final_record:
            final_record["meta"] = q.get("meta", {})
            ret["result"] = final_record

    except Exception as e:
        print(f"[Error] Processing question failed: {e}")
        
    return ret

def run_inner_loop(n_questions=20, out_dir="outputs/round_tmp", model_spec="local::", round_idx=0, max_refine=3, max_workers=20,config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            CFG = yaml.safe_load(f)
    else:
        print(f"[Warning] Config file {config_path} not found! Using empty dict.")
        CFG = {}
    os.makedirs(out_dir, exist_ok=True)
    
    critic_cfg = CFG.get("default", {}).get("critic", {})
    oracle_model = critic_cfg.get("openai_model", "gpt-4.1")
    warmup_rounds = critic_cfg.get("warmup_rounds", 0)
    
    print(f"[InnerLoop] Round {round_idx}. Oracle: {oracle_model}. Warmup Ends: {warmup_rounds}")
    print(f"[InnerLoop] Generating {n_questions} questions...")
    
    qs = generate_questions(n_questions, model_spec=model_spec, max_workers=max_workers)
    
    if not qs:
        print("[Error] No questions generated!")
        return

    results = []
    answers_pairs = []
    questions_candidates = []
    critic_pairs = [] 

    print(f"[InnerLoop] Processing questions in parallel (workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_q = {
            executor.submit(
                process_single_question, 
                q, model_spec, oracle_model, max_refine, round_idx, warmup_rounds
            ): q for q in qs
        }
        
        for future in tqdm(as_completed(future_to_q), total=len(qs), desc="Processing"):
            try:
                data = future.result()
                if data["result"]: results.append(data["result"])
                if data["answer_pair"]: answers_pairs.append(data["answer_pair"])
                # question_candidate 包含: question, label, prompt
                if data["question_candidate"]: questions_candidates.append(data["question_candidate"])
                if data["critic_pair"]: critic_pairs.append(data["critic_pair"])
            except Exception as e:
                tqdm.write(f"  Worker exception: {e}")

    # ========================================================
    # [核心修改] 构建 Question Pairs (Strictly Grouped by Prompt)
    # ========================================================
    questions_pairs = []
    
    # 1. 使用字典按 prompt 分组
    # 结构: { "prompt_text_A": {"hard": [], "easy": []}, "prompt_text_B": ... }
    grouped_candidates = defaultdict(lambda: {"hard": [], "easy": []})
    

    for item in questions_candidates:
        p = item["prompt"]
        l = item["label"] # "hard" or "easy"
        grouped_candidates[p][l].append(item["question"])
    
    # 2. 组内配对
    print(f"[Pairing] Categorizing {len(questions_candidates)} questions into {len(grouped_candidates)} unique prompts...")
    
    for prompt, groups in grouped_candidates.items():
        hards = groups["hard"]
        easies = groups["easy"]
        
        # 在同一个 prompt 下，取 hard 和 easy 的最小公倍数进行配对
        count = min(len(hards), len(easies))
        
        for i in range(count):
            questions_pairs.append({
                "prompt": prompt,          # 保证 Chosen/Rejected 都是基于此 Prompt 生成的
                "chosen": hards[i],        # Chosen: 较难的题目 (Initial 答错)
                "rejected": easies[i]      # Rejected: 较简单的题目 (Initial 答对)
            })
            
    print(f"[Pairing] Formed {len(questions_pairs)} valid Question DPO pairs.")
    # ========================================================

    print(f"[InnerLoop] Writing results to {out_dir}...")
    write_jsonl(os.path.join(out_dir, "inner_results.jsonl"), results)
    write_jsonl(os.path.join(out_dir, "answers_pairs.jsonl"), answers_pairs)
    write_jsonl(os.path.join(out_dir, "questions_pairs.jsonl"), questions_pairs)
    write_jsonl(os.path.join(out_dir, "critic_pairs.jsonl"), critic_pairs)

    dpo_data_dir = "dpo_data"
    os.makedirs(dpo_data_dir, exist_ok=True)
    try:
        shutil.copy(os.path.join(out_dir, "answers_pairs.jsonl"), os.path.join(dpo_data_dir, "answers_pairs.jsonl"))
        shutil.copy(os.path.join(out_dir, "questions_pairs.jsonl"), os.path.join(dpo_data_dir, "questions_pairs.jsonl"))
        shutil.copy(os.path.join(out_dir, "critic_pairs.jsonl"), os.path.join(dpo_data_dir, "critic_pairs.jsonl"))
    except Exception:
        pass

    print(f"[InnerLoop] Stats: Total={len(qs)}, Critic Pairs={len(critic_pairs)}, Answer Pairs={len(answers_pairs)}, Question Pairs={len(questions_pairs)}")
    return {"results": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/round_tmp")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--model_spec", type=str, default=None)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--max_refine", type=int, default=3)
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers count") 
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    ms = args.model_spec or os.environ.get("CURRENT_MODEL") or "local::/path/to/model"
    run_inner_loop(args.n_questions, args.out_dir, ms, args.round, args.max_refine, args.workers, config_path=args.config)