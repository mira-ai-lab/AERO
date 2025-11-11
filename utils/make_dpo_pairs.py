import os, json
from utils.io import read_jsonl

def convert_pairs_to_sharegpt(infile, outfile):
    recs = read_jsonl(infile)
    out = []
    for r in recs:
        prompt = r.get("question") or r.get("prompt") or ""
        
        # 转换为 LLaMA-Factory DPO (ShareGPT) 格式
        out.append({
            "conversations": [{"from": "human", "value": prompt}],
            "chosen": {"from": "gpt", "value": r["chosen"]},
            "rejected": {"from": "gpt", "value": r["rejected"]}
        })
    
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(out)} records to ShareGPT format in {outfile}")

if __name__ == "__main__":
    os.makedirs("dpo_data", exist_ok=True)
    
    if os.path.exists("dpo_data/answers_pairs.jsonl"):
        convert_pairs_to_sharegpt("dpo_data/answers_pairs.jsonl", "dpo_data/answers_dpo.json")
    
    if os.path.exists("dpo_data/questions_pairs.jsonl"):
        convert_pairs_to_sharegpt("dpo_data/questions_pairs.jsonl", "dpo_data/questions_dpo.json")