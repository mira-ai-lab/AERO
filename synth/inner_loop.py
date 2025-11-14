# synth/inner_loop.py
import argparse, os, json, random
from synth.generator import generate_questions
from synth.answerer import answer_question, self_refine
from synth.critic import verify_answer
from utils.io import write_jsonl, read_text, write_text

def run_inner_loop(n_questions=20, out_dir="outputs/round_tmp", model_spec="local::/path/to/model", max_refine=2):
    os.makedirs(out_dir, exist_ok=True)
    qs = generate_questions(n_questions, model_spec=model_spec)
    results = []
    answers_pairs = []
    questions_candidates = []

    for q in qs:
        question_text = q["question"]
        initial = answer_question(question_text, model_spec=model_spec)
        critic_report = verify_answer(initial, question_text)
        status = "passed_initial" if critic_report["passed"] else "failed_initial"
        final = initial
        if not critic_report["passed"]:
            for i in range(max_refine):
                feedback = "\n".join(critic_report.get("issues", [])[:3]) or "请修正计算或逻辑。"
                final = self_refine(question_text, final, feedback, model_spec=model_spec)
                critic_report = verify_answer(final, question_text)
                if critic_report["passed"]:
                    status = f"refined_pass_{i+1}"
                    break
            else:
                status = "failed_after_refine"

        rec = {
            "question": question_text,
            "meta": q.get("meta", {}),
            "initial_answer": initial,
            "final_answer": final,
            "status": status,
            "critic_report": critic_report
        }
        results.append(rec)

        if status.startswith("refined_pass") or status == "passed_initial":
            if status != "passed_initial":
                answers_pairs.append({"question": question_text, "chosen": final, "rejected": initial})
            if status.startswith("refined_pass") or status == "failed_after_refine":
                questions_candidates.append({"question": question_text, "label":"hard"})
            else:
                questions_candidates.append({"question": question_text, "label":"easy"})
        else:
            questions_candidates.append({"question": question_text, "label":"hard"})

    # Build question pairs (hard vs easy)
    hard = [q for q in questions_candidates if q["label"]=="hard"]
    easy = [q for q in questions_candidates if q["label"]=="easy"]
    questions_pairs = []
    m = min(len(hard), len(easy))
    for i in range(m):
        questions_pairs.append({"chosen": hard[i]["question"], "rejected": easy[i]["question"]})

    write_jsonl(os.path.join(out_dir, "inner_results.jsonl"), results)
    write_jsonl(os.path.join(out_dir, "answers_pairs.jsonl"), answers_pairs)
    write_jsonl(os.path.join(out_dir, "questions_pairs.jsonl"), questions_pairs)

    # copy to dpo_data
    os.makedirs("dpo_data", exist_ok=True)
    import shutil
    shutil.copy(os.path.join(out_dir, "answers_pairs.jsonl"), "dpo_data/answers_pairs.jsonl")
    shutil.copy(os.path.join(out_dir, "questions_pairs.jsonl"), "dpo_data/questions_pairs.jsonl")

    return {"results": results, "answers_pairs": answers_pairs, "questions_pairs": questions_pairs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/round_tmp")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--model_spec", type=str, default=None)
    args = parser.parse_args()
    ms = args.model_spec or os.environ.get("CURRENT_MODEL") or "local::/path/to/model"
    run_inner_loop(args.n_questions, out_dir=args.out_dir, model_spec=ms)
