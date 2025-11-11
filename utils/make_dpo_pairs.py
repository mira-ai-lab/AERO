# dpo/make_dpo_pairs.py
import os, json
from utils.io import read_jsonl, write_jsonl

def convert_pairs(infile, outfile):
    recs = read_jsonl(infile)
    out = []
    for r in recs:
        prompt = r.get("question") or r.get("prompt") or ""
        out.append({"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]})
    write_jsonl(outfile, out)

if __name__ == "__main__":
    os.makedirs("dpo_data", exist_ok=True)
    if os.path.exists("dpo_data/answers_pairs.jsonl"):
        convert_pairs("dpo_data/answers_pairs.jsonl", "dpo_data/answers_dpo.jsonl")
    if os.path.exists("dpo_data/questions_pairs.jsonl"):
        convert_pairs("dpo_data/questions_pairs.jsonl", "dpo_data/questions_dpo.jsonl")
    # 合并
    combined = []
    for p in ["dpo_data/answers_dpo.jsonl", "dpo_data/questions_dpo.jsonl"]:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                combined.extend([l for l in f if l.strip()])
    with open("dpo_data/combined_dpo.jsonl", 'w', encoding='utf-8') as f:
        for line in combined:
            f.write(line)
