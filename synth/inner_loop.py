# synth/inner_loop.py
import argparse
import os
import json
import shutil
import yaml
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
    返回：
      - final_answer: 最终答案
      - initial_critic_res: 对初始答案(S0)的批评结果 (用于构建 DPO)
      - trajectory: 过程记录
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
            # 自我修正 (使用 Local Answerer)
            # 注意：self_refine 通常总是使用当前正在训练的模型(Local)来改写答案
            # critic_model_spec 仅用于生成反馈
            refiner_spec = os.environ.get("CURRENT_MODEL") or critic_model_spec 
            current_answer = self_refine(question, current_answer, feedback, model_spec=refiner_spec)
            
            # 再次验证 (使用同一个 Critic)
            critic_res = verify_answer(current_answer, question, model_spec=critic_model_spec)
            trajectory.append({"step": i+1, "feedback": feedback, "answer": current_answer, "passed": critic_res["passed"]})
            
            if critic_res["passed"]:
                status = f"refined_pass_{i+1}"
                break
        else:
            status = "failed_after_refine"
            
    return {
        "final_answer": current_answer,
        "initial_critic_res": initial_critic_res, # 保留对 S0 的批评，这是 Critic DPO 的核心
        "status": status,
        "trajectory": trajectory
    }

def run_inner_loop(n_questions=20, out_dir="outputs/round_tmp", model_spec="local::/path/to/model", round_idx=0, max_refine=3):
    os.makedirs(out_dir, exist_ok=True)
    
    # 配置
    critic_cfg = CFG.get("default", {}).get("critic", {})
    oracle_model = critic_cfg.get("openai_model", "gpt-4.1")
    # 即使在 Self-Critic 阶段，我们也需要 Oracle 来做“最终裁判”和“生成正样本”
    
    print(f"[InnerLoop] Round {round_idx}. Active Model: {model_spec}")
    
    qs = generate_questions(n_questions, model_spec=model_spec)
    
    results = []
    answers_pairs = []
    questions_candidates = []
    critic_pairs = [] 

    for q in qs:
        question_text = q["question"]
        question_prompt = q.get("prompt", "")
        
        # 1. 初始作答 (S0)
        initial_answer = answer_question(question_text, model_spec=model_spec)
        
        # 2. 【真值判断】S0 是否正确？(使用 Oracle 作为客观裁判)
        #    注意：这一步是为了打标签，不参与 feedback loop
        s0_check = verify_answer(initial_answer, question_text, model_spec=oracle_model)
        s0_is_correct = s0_check["passed"]
        
        final_record = None # 用于保存到 results 的记录
        
        if s0_is_correct:
            # --- Case A: S0 初始即正确 (Easy) ---
            status = "passed_initial"
            final_record = {
                "question": question_text, "initial_answer": initial_answer, 
                "final_answer": initial_answer, "status": status, "critic_report": s0_check["critic_details"]
            }
            # 这种情况下没有 Critic 改进空间，不生成 Critic Pair
            
        else:
            # --- Case B: S0 错误 (Hard / Needs Correction) ---
            
            # 2.1 运行 Local Critic 循环
            #     让当前模型尝试自己发现并修正错误
            local_run = run_refinement_loop(question_text, initial_answer, critic_model_spec=model_spec, max_refine=max_refine)
            
            # 2.2 【真值判断】Local 修正后的最终答案是否正确？
            local_final_check = verify_answer(local_run["final_answer"], question_text, model_spec=oracle_model)
            local_success = local_final_check["passed"]
            
            if local_success:
                # -> Local 成功修正了错误。
                # 虽然这是好事，但由于我们没有“负样本”（没有失败的对照组），无法构建 Critic Pair。
                # 直接保存 Local 的成功轨迹用于 Answerer 训练。
                final_record = {
                    "question": question_text, "initial_answer": initial_answer,
                    "final_answer": local_run["final_answer"], "status": local_run["status"], "critic_report": local_run["initial_critic_res"]["critic_details"]
                }
                # Answerer DPO: S_final (Correct) > S_0 (Wrong)
                answers_pairs.append({"question": question_text, "chosen": local_run["final_answer"], "rejected": initial_answer})
                
            else:
                # -> Local 失败了 (S0 Wrong -> S_final Wrong)
                # 这是潜在的【负样本】(Rejected Critic)。
                # 为了构建 Pair，我们需要一个【正样本】，即能成功修正这个错误的 Critic。
                # 因此，此时我们需要召唤 Oracle。
                
                print(f"  [Correction Needed] Local failed on Q. Invoking Oracle...")
                oracle_run = run_refinement_loop(question_text, initial_answer, critic_model_spec=oracle_model, max_refine=max_refine)
                
                # 【真值判断】Oracle 修正后的结果是否正确？
                oracle_final_check = verify_answer(oracle_run["final_answer"], question_text, model_spec=oracle_model)
                oracle_success = oracle_final_check["passed"]
                
                if oracle_success:
                    # -> Oracle 成功修正 (S0 Wrong -> S_final Right)
                    # 【完美场景】：Local Failed (Rejected), Oracle Succeeded (Chosen)
                    
                    # 1. 构建 Critic Pair
                    # Chosen: Oracle 对 S0 的批评
                    # Rejected: Local 对 S0 的批评
                    # Prompt: Critic Prompt (包含 Question 和 S0)
                    critic_pairs.append({
                        "prompt": oracle_run["initial_critic_res"]["prompt"], 
                        "chosen": oracle_run["initial_critic_res"]["raw_output"],
                        "rejected": local_run["initial_critic_res"]["raw_output"]
                    })
                    
                    # 2. 构建 Answerer Pair
                    # 使用 Oracle 跑出来的正确答案作为 Chosen，教 Answerer 怎么改
                    answers_pairs.append({"question": question_text, "chosen": oracle_run["final_answer"], "rejected": initial_answer})
                    
                    # 记录：虽然 Local 失败了，但为了数据集的丰富性，我们记录 Oracle 救回来的结果
                    final_record = {
                        "question": question_text, "initial_answer": initial_answer,
                        "final_answer": oracle_run["final_answer"], "status": "refined_pass_oracle", "critic_report": oracle_run["initial_critic_res"]["critic_details"]
                    }
                else:
                    # -> Oracle 也失败了 (题目太难或 Oracle 也不行)
                    # 两个都烂，没有 Pair。
                    final_record = {
                        "question": question_text, "initial_answer": initial_answer,
                        "final_answer": local_run["final_answer"], "status": "failed_after_refine", "critic_report": local_run["initial_critic_res"]["critic_details"]
                    }

        # 保存 results
        if final_record:
            final_record["meta"] = q.get("meta", {})
            results.append(final_record)

        # Question DPO Candidates
        # 逻辑：一次就对(s0_correct)为 Easy，否则为 Hard
        label = "easy" if s0_is_correct else "hard"
        questions_candidates.append({
            "question": question_text, 
            "label": label,
            "prompt": question_prompt # 必须带上
        })

    # 构建 Question Pairs
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
    args = parser.parse_args()
    
    ms = args.model_spec or os.environ.get("CURRENT_MODEL") or "local::/path/to/model"
    run_inner_loop(args.n_questions, args.out_dir, ms, args.round, args.max_refine)