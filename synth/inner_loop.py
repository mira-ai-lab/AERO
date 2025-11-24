# synth/inner_loop.py
import argparse
import os
import json
import shutil
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from synth.generator import generate_questions
from synth.answerer import answer_question, self_refine
from synth.critic import verify_answer
from utils.io import write_jsonl

CFG_PATH = "config.yaml"
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        CFG = yaml.safe_load(f)
else:
    CFG = {}

def run_refinement_loop(question, initial_answer, critic_model_spec, max_refine=3):
    """
    辅助函数：运行一次完整的【批评 -> 精炼】循环。
    """
    current_answer = initial_answer
    
    # 对 S0 进行批评
    initial_critic_res = verify_answer(current_answer, question, model_spec=critic_model_spec)
    
    trajectory = []
    status = "passed_initial" if initial_critic_res["passed"] else "failed_initial"
    
    if not initial_critic_res["passed"]:
        critic_res = initial_critic_res
        for i in range(max_refine):
            # 提取反馈
            feedback = "\n".join(critic_res.get("issues", [])[:3]) or "请修正计算或逻辑。"
            # 自我修正
            refiner_spec = os.environ.get("CURRENT_MODEL") or critic_model_spec 
            current_answer = self_refine(question, current_answer, feedback, model_spec=refiner_spec)
            
            # 再次验证
            critic_res = verify_answer(current_answer, question, model_spec=critic_model_spec)
            trajectory.append({"step": i+1, "feedback": feedback, "answer": current_answer, "passed": critic_res["passed"]})
            
            if critic_res["passed"]:
                status = f"refined_pass_{i+1}"
                break
        else:
            status = "failed_after_refine"
            
    return {
        "final_answer": current_answer,
        "initial_critic_res": initial_critic_res,
        "status": status,
        "trajectory": trajectory
    }

def process_single_question(q, model_spec, oracle_model, max_refine):
    """
    处理单个问题的逻辑，用于并行执行。
    返回一个字典，包含该问题产生的所有数据记录（result, answer_pair, question_candidate, critic_pair）。
    """
    question_text = q["question"]
    question_prompt = q.get("prompt", "")
    
    # 容器初始化
    ret = {
        "result": None,
        "answer_pair": None,
        "question_candidate": None,
        "critic_pair": None
    }

    try:
        # 1. 初始作答 (S0)
        initial_answer = answer_question(question_text, model_spec=model_spec)
        
        # 2. 【真值判断】S0 是否正确？(使用 Oracle 作为客观裁判)
        s0_check = verify_answer(initial_answer, question_text, model_spec=oracle_model)
        s0_is_correct = s0_check["passed"]
        
        final_record = None
        
        if s0_is_correct:
            # --- Case A: S0 初始即正确 (Easy) ---
            status = "passed_initial"
            final_record = {
                "question": question_text, "initial_answer": initial_answer, 
                "final_answer": initial_answer, "status": status, "critic_report": s0_check["critic_details"]
            }
        else:
            # --- Case B: S0 错误 (Hard / Needs Correction) ---
            
            # 2.1 运行 Local Critic 循环
            local_run = run_refinement_loop(question_text, initial_answer, critic_model_spec=model_spec, max_refine=max_refine)
            
            # 2.2 【真值判断】Local 修正后的最终答案是否正确？
            local_final_check = verify_answer(local_run["final_answer"], question_text, model_spec=oracle_model)
            local_success = local_final_check["passed"]
            
            if local_success:
                # -> Local 成功修正了错误
                final_record = {
                    "question": question_text, "initial_answer": initial_answer,
                    "final_answer": local_run["final_answer"], "status": local_run["status"], "critic_report": local_run["initial_critic_res"]["critic_details"]
                }
                ret["answer_pair"] = {"question": question_text, "chosen": local_run["final_answer"], "rejected": initial_answer}
                
            else:
                # -> Local 失败了，尝试召唤 Oracle
                # print(f"  [Correction Needed] Local failed on Q. Invoking Oracle...") # 多线程下减少 print 防止刷屏
                oracle_run = run_refinement_loop(question_text, initial_answer, critic_model_spec=oracle_model, max_refine=max_refine)
                
                # 【真值判断】Oracle 修正后的结果是否正确？
                oracle_final_check = verify_answer(oracle_run["final_answer"], question_text, model_spec=oracle_model)
                oracle_success = oracle_final_check["passed"]
                
                if oracle_success:
                    # -> Oracle 成功修正
                    # 1. 构建 Critic Pair
                    ret["critic_pair"] = {
                        "prompt": oracle_run["initial_critic_res"]["prompt"], 
                        "chosen": oracle_run["initial_critic_res"]["raw_output"],
                        "rejected": local_run["initial_critic_res"]["raw_output"]
                    }
                    
                    # 2. 构建 Answerer Pair
                    ret["answer_pair"] = {"question": question_text, "chosen": oracle_run["final_answer"], "rejected": initial_answer}
                    
                    final_record = {
                        "question": question_text, "initial_answer": initial_answer,
                        "final_answer": oracle_run["final_answer"], "status": "refined_pass_oracle", "critic_report": oracle_run["initial_critic_res"]["critic_details"]
                    }
                else:
                    # -> Oracle 也失败了
                    final_record = {
                        "question": question_text, "initial_answer": initial_answer,
                        "final_answer": local_run["final_answer"], "status": "failed_after_refine", "critic_report": local_run["initial_critic_res"]["critic_details"]
                    }

        if final_record:
            final_record["meta"] = q.get("meta", {})
            ret["result"] = final_record

        # Question DPO Candidates
        label = "easy" if s0_is_correct else "hard"
        ret["question_candidate"] = {
            "question": question_text, 
            "label": label,
            "prompt": question_prompt
        }

    except Exception as e:
        print(f"[Error] Processing question failed: {e}")
        
    return ret

def run_inner_loop(n_questions=20, out_dir="outputs/round_tmp", model_spec="local::/path/to/model", round_idx=0, max_refine=3, max_workers=10):
    os.makedirs(out_dir, exist_ok=True)
    
    # 配置
    critic_cfg = CFG.get("default", {}).get("critic", {})
    oracle_model = critic_cfg.get("openai_model", "gpt-4.1")
    
    print(f"[InnerLoop] Round {round_idx}. Active Model: {model_spec}")
    print(f"[InnerLoop] Generating {n_questions} questions...")
    
    qs = generate_questions(n_questions, model_spec=model_spec)
    
    results = []
    answers_pairs = []
    questions_candidates = []
    critic_pairs = [] 

    print(f"[InnerLoop] Processing questions in parallel (workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_q = {executor.submit(process_single_question, q, model_spec, oracle_model, max_refine): q for q in qs}
        
        count = 0
        total = len(qs)
        
        for future in as_completed(future_to_q):
            count += 1
            if count % 5 == 0:
                print(f"  Progress: {count}/{total}")
                
            try:
                data = future.result()
                if data["result"]: results.append(data["result"])
                if data["answer_pair"]: answers_pairs.append(data["answer_pair"])
                if data["question_candidate"]: questions_candidates.append(data["question_candidate"])
                if data["critic_pair"]: critic_pairs.append(data["critic_pair"])
            except Exception as e:
                print(f"  Worker exception: {e}")

    # 构建 Question Pairs (逻辑保持不变)
    hard_qs = [x for x in questions_candidates if x["label"] == "hard"]
    easy_qs = [x for x in questions_candidates if x["label"] == "easy"]
    m = min(len(hard_qs), len(easy_qs))
    questions_pairs = []
    for i in range(m):
        questions_pairs.append({
            "prompt": hard_qs[i]["prompt"],
            "chosen": hard_qs[i]["question"],
            "rejected": easy_qs[i]["question"]
        })

    # 写入文件
    print(f"[InnerLoop] Writing results to {out_dir}...")
    write_jsonl(os.path.join(out_dir, "inner_results.jsonl"), results)
    write_jsonl(os.path.join(out_dir, "answers_pairs.jsonl"), answers_pairs)
    write_jsonl(os.path.join(out_dir, "questions_pairs.jsonl"), questions_pairs)
    write_jsonl(os.path.join(out_dir, "critic_pairs.jsonl"), critic_pairs)

    # 复制到 dpo_data
    dpo_data_dir = "dpo_data"
    os.makedirs(dpo_data_dir, exist_ok=True)
    try:
        shutil.copy(os.path.join(out_dir, "answers_pairs.jsonl"), os.path.join(dpo_data_dir, "answers_pairs.jsonl"))
        shutil.copy(os.path.join(out_dir, "questions_pairs.jsonl"), os.path.join(dpo_data_dir, "questions_pairs.jsonl"))
        shutil.copy(os.path.join(out_dir, "critic_pairs.jsonl"), os.path.join(dpo_data_dir, "critic_pairs.jsonl"))
    except Exception:
        pass

    print(f"[InnerLoop] Stats: Total={len(qs)}, Critic Pairs={len(critic_pairs)}, Answer Pairs={len(answers_pairs)}")
    return {"results": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/round_tmp")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--model_spec", type=str, default=None)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--max_refine", type=int, default=3)
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers count") # 新增参数
    args = parser.parse_args()
    
    ms = args.model_spec or os.environ.get("CURRENT_MODEL") or "local::/path/to/model"
    run_inner_loop(args.n_questions, args.out_dir, ms, args.round, args.max_refine, args.workers)